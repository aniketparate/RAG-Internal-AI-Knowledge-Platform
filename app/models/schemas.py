"""Pydantic schemas for API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


DocumentStatus = Literal["pending", "processing", "ready", "failed", "partial"]


class DocumentCreateResponse(BaseModel):
    """Response for queued ingestion jobs."""

    document_id: UUID
    job_id: str
    status: DocumentStatus
    message: str


class DocumentResponse(BaseModel):
    """Document metadata and current status."""

    document_id: UUID
    file_name: str
    file_type: str
    status: DocumentStatus
    chunk_count: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteDocumentResponse(BaseModel):
    """Delete status payload."""

    document_id: UUID
    deleted: bool
    hard_delete: bool


class QueryFilters(BaseModel):
    """Optional filters applied to vector retrieval."""

    file_type: str | None = None
    document_id: UUID | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class QueryRequest(BaseModel):
    """Query request payload."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: QueryFilters | None = None


class QueryResult(BaseModel):
    """Single retrieval result."""

    chunk_id: UUID
    document_id: UUID
    file_name: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Query response payload."""

    query: str
    results: list[QueryResult]
    latency_ms: int


class ChatMessage(BaseModel):
    """Message item for chat payload."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Gateway chat request payload."""

    messages: list[ChatMessage]
    model: str
    rag_query: str | None = None
    top_k: int | None = Field(default=3, ge=1, le=20)


class ChatResponse(BaseModel):
    """Standardized gateway response."""

    model: str
    content: str
    raw_response: dict[str, Any] | None = None
