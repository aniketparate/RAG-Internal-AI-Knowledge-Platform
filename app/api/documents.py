"""Document ingestion and lifecycle API endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.models.db import Document
from app.models.schemas import DeleteDocumentResponse, DocumentCreateResponse, DocumentResponse
from app.services.vector_store import VectorStore
from app.tasks.celery_app import ingest_document_task

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {
    "pdf",
    "md",
    "txt",
    "py",
    "js",
    "ts",
    "go",
    "java",
    "docx",
}


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentCreateResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
) -> DocumentCreateResponse:

    file_name = file.filename or "uploaded_file"
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    metadata_payload: dict = {}
    if metadata:
        try:
            metadata_payload = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

    # Read file BEFORE touching the DB session
    file_bytes = await file.read()

    settings = get_settings()
    uploads_dir = Path(settings.upload_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    document = Document(
        file_name=file_name,
        file_type=extension,
        file_size_bytes=len(file_bytes),  # set it here, not after
        status="pending",
        metadata_json=metadata_payload,
    )
    db.add(document)
    await db.commit()          # single commit
    await db.refresh(document)

    # Save file after commit — document.id is now available
    file_path = uploads_dir / f"{document.id}_{file_name}"
    async with aiofiles.open(file_path, mode="wb") as f:
        await f.write(file_bytes)

    task = ingest_document_task.delay(str(document.id), str(file_path), extension)
    return DocumentCreateResponse(
        document_id=document.id,
        job_id=task.id,
        status="pending",
        message="Document queued for processing",
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID, db: AsyncSession = Depends(get_db)) -> DocumentResponse:
    """Get document processing status and metadata."""

    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        document_id=document.id,
        file_name=document.file_name,
        file_type=document.file_type,
        status=document.status,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        metadata=document.metadata_json or {},
    )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: UUID,
    hard: bool = False,
    db: AsyncSession = Depends(get_db),
) -> DeleteDocumentResponse:
    """Soft-delete or hard-delete a document and vector payloads."""

    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    settings = get_settings()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = VectorStore(client=client, collection=settings.qdrant_collection)

    try:
        if hard:
            failures: list[str] = []
            try:
                await store.delete_by_document(str(document_id), hard_delete=True)
            except Exception as exc:
                logger.exception("Hard delete failed in Qdrant for document_id=%s", document_id)
                failures.append(f"qdrant:{exc}")
            try:
                await db.delete(document)
                await db.commit()
            except Exception as exc:
                logger.exception("Hard delete failed in DB for document_id=%s", document_id)
                failures.append(f"database:{exc}")
                await db.rollback()

            if failures:
                raise HTTPException(status_code=500, detail=f"Hard delete partially failed: {failures}")
            return DeleteDocumentResponse(document_id=document_id, deleted=True, hard_delete=True)

        await store.delete_by_document(str(document_id), hard_delete=False)
        document.deleted_at = datetime.now(timezone.utc)
        await db.commit()
        return DeleteDocumentResponse(document_id=document_id, deleted=True, hard_delete=False)
    finally:
        await client.close()
