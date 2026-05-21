"""Shared pytest fixtures for integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return repository root path."""

    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_url() -> str:
    """Return integration test API base URL."""

    return os.getenv("TEST_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_key() -> str:
    """Return API key used by tests."""

    return os.getenv("API_KEY", "internal-dev-key")


@pytest.fixture
async def api_client(base_url: str, api_key: str) -> AsyncClient:
    """Create an HTTP client targeting the running API container."""

    async with AsyncClient(base_url=base_url, headers={"X-API-Key": api_key}, timeout=120.0) as client:
        yield client
