"""FastAPI application entrypoint."""

from __future__ import annotations

import json
import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.api.documents import router as documents_router
from app.api.gateway import router as gateway_router
from app.api.query import router as query_router
from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("rag_platform")


async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Validate internal API key."""

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(
    title="Internal AI Knowledge Platform",
    version="1.0.0",
    description="Production-grade RAG backend for internal developer tooling",
    dependencies=[Depends(require_api_key)],
)


@app.middleware("http")
async def json_logging_middleware(request: Request, call_next):
    """Emit structured JSON logs for each HTTP request."""

    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(
            json.dumps(
                {
                    "event": "request_error",
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                }
            ),
            exc_info=True,
        )
        raise

    latency_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        json.dumps(
            {
                "event": "request_complete",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        )
    )
    return response


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize shared services and ensure vector collection exists."""

    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collections = await qdrant_client.get_collections()
    existing = {collection.name for collection in collections.collections}
    if settings.qdrant_collection not in existing:
        await qdrant_client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_vector_size, distance=Distance.COSINE),
        )
    else:
        collection_info = await qdrant_client.get_collection(settings.qdrant_collection)
        vector_cfg = collection_info.config.params.vectors
        current_size = vector_cfg.size if hasattr(vector_cfg, "size") else None
        if current_size != settings.embedding_vector_size:
            if settings.qdrant_recreate_on_vector_mismatch:
                logger.warning(
                    "Recreating Qdrant collection '%s' due to vector size mismatch (expected=%s actual=%s)",
                    settings.qdrant_collection,
                    settings.embedding_vector_size,
                    current_size,
                )
                await qdrant_client.delete_collection(settings.qdrant_collection)
                await qdrant_client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=VectorParams(size=settings.embedding_vector_size, distance=Distance.COSINE),
                )
            else:
                raise RuntimeError(
                    "Qdrant vector size mismatch for collection "
                    f"'{settings.qdrant_collection}': expected={settings.embedding_vector_size}, actual={current_size}. "
                    "Delete/recreate the collection or set QDRANT_RECREATE_ON_VECTOR_MISMATCH=true."
                )

    app.state.qdrant_client = qdrant_client
    app.state.vector_store = VectorStore(client=qdrant_client, collection=settings.qdrant_collection)
    app.state.embedding_service = EmbeddingService()
    app.state.reranker = Reranker(settings.reranker_model)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Release external clients."""

    client: AsyncQdrantClient = app.state.qdrant_client
    await client.close()


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Standardize HTTP exceptions while preserving FastAPI validation behavior."""

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(documents_router)
app.include_router(query_router)
app.include_router(gateway_router)
