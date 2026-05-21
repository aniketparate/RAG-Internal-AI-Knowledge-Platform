"""Asynchronous ingestion orchestrator."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db import Chunk as ChunkModel
from app.models.db import Document
from app.services.chunking import Chunk, ChunkingService
from app.services.embedding import EmbeddingService
from app.services.extraction import ContentExtractor
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def ingest_document(db: AsyncSession, document_id: str, file_path: str, file_type: str) -> None:
    """Run extract -> chunk -> embed -> upsert pipeline for a document."""

    settings = get_settings()
    extractor = ContentExtractor()
    chunker = ChunkingService()
    embedding_service = EmbeddingService()
    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    vector_store = VectorStore(client=qdrant_client, collection=settings.qdrant_collection)
    normalized_file_type = file_type.lower()

    try:
        document = await _set_document_status(db, UUID(document_id), "processing")
        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        logger.info("Starting ingestion for document_id=%s", document_id)
        content = await extractor.extract(file_path=file_path, file_type=normalized_file_type)
        base_metadata = {
            "file_name": document.file_name,
            "file_type": normalized_file_type,
            "page_number": None,
            "function_name": None,
            "class_name": None,
        }
        chunks = chunker.chunk_content(content=content, file_type=normalized_file_type, base_metadata=base_metadata)
        if not chunks:
            raise ValueError("No chunks generated for document")
        logger.info(
            "Chunking completed document_id=%s file_type=%s chunk_count=%s token_count=%s",
            document_id,
            normalized_file_type,
            len(chunks),
            chunker.token_count(content),
        )

        successful_chunks, vectors, failed_indices = await _embed_chunks_with_partial(chunks, embedding_service)
        if not successful_chunks:
            raise RuntimeError("All chunk embeddings failed")

        point_ids = await vector_store.upsert_chunks(
            document_id=document_id,
            chunks=successful_chunks,
            vectors=vectors,
        )
        await _store_chunks(db, UUID(document_id), successful_chunks, point_ids, chunker)

        if failed_indices:
            message = f"Partial chunk embedding failures at indices: {failed_indices}"
            await _set_document_status(db, UUID(document_id), "partial", len(successful_chunks), message)
            logger.info("Completed ingestion with partial failures document_id=%s", document_id)
        else:
            await _set_document_status(db, UUID(document_id), "ready", len(successful_chunks), None)
            logger.info("Completed ingestion document_id=%s", document_id)
    except Exception as exc:
        logger.exception("Ingestion failed document_id=%s", document_id)
        await db.rollback()
        await _set_document_status(db, UUID(document_id), "failed", error_message=str(exc))
        raise
    finally:
        await qdrant_client.close()


async def _embed_chunks_with_partial(
    chunks: list[Chunk], embedding_service: EmbeddingService
) -> tuple[list[Chunk], list[list[float]], list[int]]:
    """Embed chunks and keep successful items when partial failures occur."""

    try:
        vectors = await embedding_service.embed_texts([chunk.content for chunk in chunks])
        return chunks, vectors, []
    except Exception:
        successful_chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        failed_indices: list[int] = []
        for chunk in chunks:
            try:
                vector = await embedding_service.embed_query(chunk.content)
                successful_chunks.append(chunk)
                vectors.append(vector)
            except Exception:
                failed_indices.append(chunk.chunk_index)
        return successful_chunks, vectors, failed_indices


async def _set_document_status(
    session: AsyncSession,
    document_id: UUID,
    status: str,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> Document | None:
    """Update and persist a document status."""

    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        await session.commit()
        return None
    document.status = status
    if chunk_count is not None:
        document.chunk_count = chunk_count
    document.error_message = error_message
    await session.commit()
    await session.refresh(document)
    return document


async def _store_chunks(
    session: AsyncSession,
    document_id: UUID,
    chunks: list[Chunk],
    point_ids: list[Any],
    chunker: ChunkingService,
) -> None:
    """Persist chunk metadata rows in PostgreSQL."""

    for chunk, point_id in zip(chunks, point_ids, strict=True):
        session.add(
            ChunkModel(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunker.token_count(chunk.content),
                qdrant_point_id=point_id,
                metadata_json=chunk.metadata,
            )
        )
    await session.commit()
