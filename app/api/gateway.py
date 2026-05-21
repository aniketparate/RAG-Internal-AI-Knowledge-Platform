"""LLM gateway endpoint with optional RAG context injection."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.models.schemas import ChatMessage, ChatRequest, ChatResponse
from app.services.embedding import EmbeddingService
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

router = APIRouter(tags=["gateway"])


@router.post("/gateway/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Proxy chat requests with optional retrieval context."""

    settings = get_settings()
    system_context = ""
    if body.rag_query:
        chunks = await _retrieve_context(request, body.rag_query, body.top_k or 3)
        system_context = _format_context(chunks)

    if _should_use_groq(settings, body.model):
        groq_key = settings.groq_api_key or settings.xai_api_key
        return await _call_groq(
            body,
            groq_key,
            settings.llm_base_url_groq,
            settings.llm_default_model_groq,
            system_context,
        )

    if _should_use_xai(settings, body.model):
        return await _call_xai(body, settings.xai_api_key, settings.llm_base_url_xai, system_context)

    if body.model.lower().startswith("claude"):
        if not settings.anthropic_api_key:
            raise HTTPException(status_code=400, detail="Anthropic API key is not configured")
        return await _call_anthropic(body, settings.anthropic_api_key, system_context)

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="No valid LLM provider key configured. Set GROQ_API_KEY or OPENAI_API_KEY.",
        )
    return await _call_openai(body, settings.openai_api_key, system_context)


async def _retrieve_context(request: Request, query: str, top_k: int) -> list[dict[str, Any]]:
    """Run retrieval pipeline and return lightweight context chunks."""

    embedding_service: EmbeddingService = request.app.state.embedding_service
    vector_store: VectorStore = request.app.state.vector_store
    reranker: Reranker = request.app.state.reranker

    query_vector = await embedding_service.embed_query(query)
    candidates = await vector_store.search(query_vector, top_k * 4, filters=None)
    candidates = [candidate for candidate in candidates if not candidate.payload.get("deleted")]
    ranked = reranker.rerank(query, candidates, top_k)

    results: list[dict[str, Any]] = []
    for item in ranked:
        payload = item.point.payload
        results.append(
            {
                "chunk_id": str(item.point.id),
                "document_id": str(payload.get("document_id", "")),
                "file_name": str(payload.get("file_name", "")),
                "content": str(payload.get("content", "")),
                "score": item.score,
            }
        )
    return results


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Create system-context prompt text from retrieval chunks."""

    if not chunks:
        return ""
    lines = ["Use the following retrieved context when answering:"]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[{idx}] file={chunk['file_name']} score={chunk['score']:.4f}")
        lines.append(chunk["content"])
        lines.append("")
    return "\n".join(lines).strip()


def _should_use_xai(settings, model: str) -> bool:
    """Resolve whether to route this request to xAI."""

    if model.lower().startswith("grok"):
        return True
    if settings.llm_provider.lower() == "xai" and bool(settings.xai_api_key):
        return True
    return False


def _should_use_groq(settings, model: str) -> bool:
    """Resolve whether to route this request to Groq."""

    provider = settings.llm_provider.lower()
    has_groq_key = bool(settings.groq_api_key or settings.xai_api_key)
    if provider == "groq" and has_groq_key:
        return True
    normalized = model.lower()
    if normalized.startswith(("llama", "deepseek", "openai/", "mixtral", "qwen", "gemma")) and has_groq_key:
        return True
    return False


async def _call_groq(
    body: ChatRequest,
    api_key: str,
    base_url: str,
    default_model: str,
    system_context: str,
) -> ChatResponse:
    """Forward chat request to Groq OpenAI-compatible Chat Completions."""

    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ API key is not configured")

    messages = _sanitize_openai_messages(body.messages)
    if system_context:
        messages = [{"role": "system", "content": system_context}, *messages]
    routed_model = _resolve_groq_model(body.model, default_model)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": routed_model, "messages": messages},
        )

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Groq API error: {response.text}")
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return ChatResponse(model=routed_model, content=content, raw_response=payload)


async def _call_xai(body: ChatRequest, api_key: str, base_url: str, system_context: str) -> ChatResponse:
    """Forward chat request to xAI Chat Completions."""

    if not api_key:
        raise HTTPException(status_code=400, detail="XAI API key is not configured")

    messages = _sanitize_openai_messages(body.messages)
    if system_context:
        messages = [{"role": "system", "content": system_context}, *messages]
    routed_model = _resolve_xai_model(body.model)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": routed_model, "messages": messages},
        )

    print("xAI error body:", response.text)
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return ChatResponse(model=routed_model, content=content, raw_response=payload)


async def _call_openai(body: ChatRequest, api_key: str, system_context: str) -> ChatResponse:
    """Forward chat request to OpenAI Chat Completions."""

    messages = [message.model_dump() for message in body.messages]
    if system_context:
        messages = [{"role": "system", "content": system_context}, *messages]

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": body.model, "messages": messages},
        )

    print("OpenAI error body:", response.text)
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return ChatResponse(model=body.model, content=content, raw_response=payload)


async def _call_anthropic(body: ChatRequest, api_key: str, system_context: str) -> ChatResponse:
    """Forward chat request to Anthropic Messages API."""

    anthropic_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in body.messages
        if msg.role in {"user", "assistant"}
    ]
    system_text = system_context or None

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": body.model,
                "max_tokens": 1024,
                "system": system_text,
                "messages": anthropic_messages,
            },
        )

    print("Anthropic error body:", response.text)
    response.raise_for_status()
    payload = response.json()
    content_blocks = payload.get("content", [])
    content = ""
    if content_blocks:
        content = str(content_blocks[0].get("text", ""))
    return ChatResponse(model=body.model, content=content, raw_response=payload)


def _resolve_xai_model(requested_model: str | None) -> str:
    """Map incoming model names to supported xAI routing."""

    if not requested_model:
        return "grok-3"
    normalized = requested_model.strip().lower()
    if normalized.startswith("grok-"):
        return requested_model.strip()
    if normalized == "gpt-4o":
        return "grok-3"
    return "grok-3"


def _resolve_groq_model(requested_model: str | None, default_model: str) -> str:
    """Map incoming model names to Groq-supported defaults."""

    if not requested_model:
        return default_model
    normalized = requested_model.strip().lower()
    if normalized == "gpt-4o":
        return default_model
    if normalized.startswith("grok-"):
        return default_model
    return requested_model.strip()


def _sanitize_openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Keep only OpenAI-compatible role/content message fields."""

    sanitized: list[dict[str, str]] = []
    for message in messages:
        if message.role not in {"system", "user", "assistant"}:
            continue
        sanitized.append({"role": message.role, "content": message.content})
    return sanitized
