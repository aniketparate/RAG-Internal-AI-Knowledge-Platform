"""Qdrant vector store wrapper."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, FilterSelector, MatchValue, PointStruct, Range, ScoredPoint

from app.services.chunking import Chunk


class VectorStore:
    """Async wrapper over a Qdrant collection."""

    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    async def upsert_chunks(self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> list[uuid.UUID]:
        """Upsert chunk vectors and payloads into Qdrant."""

        point_ids: list[uuid.UUID] = []
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vector_batch = vectors[start : start + batch_size]
            points: list[PointStruct] = []
            for chunk, vector in zip(chunk_batch, vector_batch, strict=True):
                point_id = uuid.uuid4()
                point_ids.append(point_id)
                payload = {
                    **chunk.metadata,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "created_at": datetime.utcnow().isoformat(),
                }
                points.append(PointStruct(id=str(point_id), vector=vector, payload=payload))
            await self._retry_upsert(points)
        return point_ids

    async def search(self, query_vector: list[float], top_k: int, filters: dict | None) -> list[ScoredPoint]:
        """Run ANN search with optional metadata filters."""

        query_filter = self._build_filter(filters)
        return await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=0.0,
        )

    async def delete_by_document(self, document_id: str, hard_delete: bool = False) -> None:
        """Delete or soft-delete points for a specific document."""

        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        if hard_delete:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=FilterSelector(filter=doc_filter),
            )
            return
        await self.client.set_payload(
            collection_name=self.collection,
            payload={"deleted": True},
            points=doc_filter,
        )

    async def _retry_upsert(self, points: list[PointStruct]) -> None:
        """Retry Qdrant upsert failures with exponential backoff."""

        attempt = 0
        backoff = 1.0
        while True:
            try:
                await self.client.upsert(collection_name=self.collection, points=points, wait=True)
                return
            except Exception as exc:
                attempt += 1
                if attempt >= 3:
                    raise RuntimeError(f"Qdrant upsert failed after retries: {exc}") from exc
                await asyncio.sleep(backoff)
                backoff *= 2

    def _build_filter(self, filters: dict | None) -> Filter | None:
        """Convert API filters into Qdrant filter expressions."""

        if not filters:
            return None

        must: list[FieldCondition] = []
        file_type = filters.get("file_type")
        document_id = filters.get("document_id")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        if file_type:
            must.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))
        if document_id:
            must.append(FieldCondition(key="document_id", match=MatchValue(value=str(document_id))))
        if date_from or date_to:
            must.append(
                FieldCondition(
                    key="created_at",
                    range=Range(gte=date_from.isoformat() if date_from else None, lte=date_to.isoformat() if date_to else None),
                )
            )
        return Filter(must=must) if must else None
