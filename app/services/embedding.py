"""Embedding client wrapper with retry and batching."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating document and query embeddings."""

    _local_model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self) -> None:
        self.settings = get_settings()
        self._provider = self._resolve_provider()
        self._remote_url, self._remote_headers = self._build_remote_client(self._provider)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in configured batches."""

        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_batch_with_fallback(batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

        vectors = await self._embed_batch_with_fallback([text])
        return vectors[0]

    async def _embed_batch_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Embed with provider preference and resilient local fallback."""

        if self._provider == "local":
            return await self._embed_local(texts)

        try:
            return await self._embed_remote(texts)
        except Exception as exc:
            # Spec originally pinned OpenAI embeddings; this fallback keeps ingestion/query operational
            # when only Grok/xAI chat keys are available or remote embedding model access differs.
            logger.warning("Remote embedding failed; falling back to local model: %s", exc)
            return await self._embed_local(texts)

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings with retry on rate-limit/server errors."""

        attempt = 0
        backoff = 1.0
        while True:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self._remote_url,
                        headers=self._remote_headers,
                        json={"model": self.settings.embedding_model, "input": texts},
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable embedding status", request=response.request, response=response)
                response.raise_for_status()
                body: dict[str, Any] = response.json()
                data = body.get("data", [])
                return [item["embedding"] for item in data]
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                attempt += 1
                if attempt >= 3:
                    raise RuntimeError(f"Embedding request failed after retries: {exc}") from exc
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with a local sentence-transformers model."""

        model = self._get_or_load_local_model()
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return vectors.tolist()

    def _resolve_provider(self) -> str:
        """Resolve embedding provider from settings and available keys."""

        provider = self.settings.embedding_provider.lower()
        if provider in {"openai", "xai", "local"}:
            return provider
        if self.settings.xai_api_key:
            return "xai"
        if self.settings.openai_api_key:
            return "openai"
        return "local"

    def _build_remote_client(self, provider: str) -> tuple[str, dict[str, str]]:
        """Build remote endpoint and auth headers for selected provider."""

        if provider == "xai":
            if not self.settings.xai_api_key:
                raise ValueError("XAI_API_KEY is required when EMBEDDING_PROVIDER=xai")
            return (
                f"{self.settings.embedding_api_base_url.rstrip('/')}/embeddings",
                {"Authorization": f"Bearer {self.settings.xai_api_key}", "Content-Type": "application/json"},
            )
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            return (
                "https://api.openai.com/v1/embeddings",
                {"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"},
            )
        return "", {}

    def _get_or_load_local_model(self) -> SentenceTransformer:
        """Load and cache a local embedding model instance."""

        model_name = self.settings.local_embedding_model
        cached = self._local_model_cache.get(model_name)
        if cached is not None:
            return cached
        model = SentenceTransformer(model_name)
        self._local_model_cache[model_name] = model
        return model
