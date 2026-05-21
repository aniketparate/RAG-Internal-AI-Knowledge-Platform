"""Query API endpoint for retrieval and reranking."""

from __future__ import annotations

import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.db import QueryLog
from app.models.schemas import QueryRequest, QueryResponse, QueryResult
from app.services.embedding import EmbeddingService
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(body: QueryRequest, request: Request, db: AsyncSession = Depends(get_db)) -> QueryResponse:
    """Run retrieval pipeline and return top-ranked chunks."""

    start = time.monotonic()
    embedding_service: EmbeddingService = request.app.state.embedding_service
    vector_store: VectorStore = request.app.state.vector_store
    reranker: Reranker = request.app.state.reranker

    query_vector = await embedding_service.embed_query(body.query)
    filters_dict = body.filters.model_dump(exclude_none=True) if body.filters else None
    candidates = await vector_store.search(query_vector, body.top_k * 4, filters_dict)
    candidates = [candidate for candidate in candidates if not candidate.payload.get("deleted")]
    ranked_results = reranker.rerank(body.query, candidates, body.top_k)

    results: list[QueryResult] = []
    for ranked in ranked_results:
        payload = ranked.point.payload
        try:
            chunk_id = UUID(str(ranked.point.id))
            document_id = UUID(str(payload["document_id"]))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Invalid retrieval payload UUIDs: {exc}") from exc
        results.append(
            QueryResult(
                chunk_id=chunk_id,
                document_id=document_id,
                file_name=str(payload.get("file_name", "")),
                content=str(payload.get("content", "")),
                score=ranked.score,
                metadata={k: v for k, v in payload.items() if k not in {"content", "document_id", "file_name"}},
            )
        )

    latency_ms = int((time.monotonic() - start) * 1000)
    retrieved_ids = list({result.document_id for result in results})
    db.add(
        QueryLog(
            query_text=body.query,
            filters=filters_dict or {},
            top_k=body.top_k,
            result_count=len(results),
            latency_ms=latency_ms,
            retrieved_document_ids=retrieved_ids,
        )
    )
    await db.commit()

    return QueryResponse(query=body.query, results=results, latency_ms=latency_ms)
