# Internal AI Knowledge Platform

A production-grade RAG backend for internal developer tooling.

## Prerequisites
- Docker and Docker Compose
- Python 3.11+

## Setup
1. Copy `.env.example` to `.env`.
2. Set `XAI_API_KEY` and keep `LLM_PROVIDER=xai`.
3. Keep `EMBEDDING_PROVIDER=local` unless you have a compatible remote embedding endpoint.
4. Start dependencies and services:
   `docker-compose up -d --build`
5. Run migrations:
   `alembic upgrade head`

Uploaded files are stored in `/tmp/rag_uploads` on a shared Docker volume so the API and Celery worker can both read them without triggering Uvicorn reloads.

If you switch embedding dimensions (for example from 1536 to 384), recreate the Qdrant collection or set `QDRANT_RECREATE_ON_VECTOR_MISMATCH=true` in `.env` for automatic recreation on startup.

## Ingesting Documents
```bash
curl -X POST http://localhost:8000/documents \
  -H "X-API-Key: internal-dev-key" \
  -F "file=@Source_Code_Sample.py"
```

```bash
curl -X POST http://localhost:8000/documents \
  -H "X-API-Key: internal-dev-key" \
  -F "file=@Knowledge_Base_Sample.pdf"
```

## Querying
```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: internal-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the proxy rotator handle failures?", "top_k": 3}'
```

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: internal-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI orchestration and why does it matter?", "top_k": 3}'
```

```bash
curl -X POST http://localhost:8000/gateway/chat \
  -H "X-API-Key: internal-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain how the DecayProxyRotator works"}],
    "model": "grok-3-mini",
    "rag_query": "DecayProxyRotator implementation",
    "top_k": 3
  }'
```

## Running Tests
`pytest tests/ -v`

## Scaling Strategy
- Horizontal API scaling: run multiple FastAPI containers behind a load balancer.
- Celery scaling: add worker replicas and tune `--concurrency`.
- Qdrant scaling: move from single-node to clustered/sharded deployment when vector count grows.
- Embedding cost control: cache embeddings by content hash in Redis.
- Query latency control: add short-lived Redis cache for repeated queries.
- Database analytics scale: add Postgres read replicas for log-heavy reads.
- Reranker scale: move cross-encoder to dedicated compute if latency rises.

## Tradeoffs
- Qdrant is self-hosted and cost-efficient, but operationally owned by the team.
- `text-embedding-3-small` is affordable; larger models can improve quality at higher cost.
- Celery adds infrastructure but provides durable async processing.
- Soft delete protects against accidental data loss but requires query-time filtering.
- Reranking improves precision with additional latency.
- No multi-tenancy is acceptable for a trusted internal audience; ACLs can be added later.
- 512-token chunking balances context breadth and retrieval specificity.
