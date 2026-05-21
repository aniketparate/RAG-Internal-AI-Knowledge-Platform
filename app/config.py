"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Internal AI Knowledge Platform."""

    database_url: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledge_base"
    redis_url: str = "redis://localhost:6379/0"
    xai_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    api_key: str
    llm_provider: str = "xai"
    llm_base_url_xai: str = "https://api.x.ai/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    embedding_provider: str = "local"
    embedding_api_base_url: str = "https://api.x.ai/v1"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_vector_size: int = 384
    qdrant_recreate_on_vector_mismatch: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    upload_dir: str = "/tmp/rag_uploads"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
