"""Celery application and task definitions."""

from __future__ import annotations

import asyncio

from celery import Celery

from app.config import get_settings
from app.db.session import AsyncSessionLocal
from app.services.ingestion import ingest_document

settings = get_settings()
celery_app = Celery("rag_platform", broker=settings.redis_url)


@celery_app.task(bind=True, max_retries=3)
def ingest_document_task(self, document_id: str, file_path: str, file_type: str) -> None:
    """Run asynchronous document ingestion inside a Celery worker."""

    asyncio.run(_run_ingestion(document_id=document_id, file_path=file_path, file_type=file_type))


async def _run_ingestion(document_id: str, file_path: str, file_type: str) -> None:
    """Create a fresh DB session inside the Celery task process."""

    async with AsyncSessionLocal() as db:
        await ingest_document(db=db, document_id=document_id, file_path=file_path, file_type=file_type)
