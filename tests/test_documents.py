"""Integration tests for document ingestion endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

from httpx import AsyncClient


async def _upload_file(client: AsyncClient, file_path: Path) -> str:
    """Upload a file and return document ID."""

    with file_path.open("rb") as file_obj:
        response = await client.post(
            "/documents",
            files={"file": (file_path.name, file_obj, "application/octet-stream")},
        )
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "pending"
    return payload["document_id"]


async def _wait_until_ready(client: AsyncClient, document_id: str, timeout_seconds: int = 180) -> dict:
    """Poll document status until terminal state."""

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/documents/{document_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"ready", "partial", "failed"}:
            return payload
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for document {document_id}")


async def test_upload_and_get_code_document(api_client: AsyncClient, project_root: Path) -> None:
    """Ingest Source_Code_Sample.py and validate status endpoint."""

    source_file = project_root / "Source_Code_Sample.py"
    document_id = await _upload_file(api_client, source_file)
    payload = await _wait_until_ready(api_client, document_id)

    assert payload["document_id"] == document_id
    assert payload["file_name"] == "Source_Code_Sample.py"
    assert payload["file_type"] in {"py", "python"}
    assert payload["status"] in {"ready", "partial"}
    assert payload["chunk_count"] >= 1


async def test_soft_delete_document(api_client: AsyncClient, project_root: Path) -> None:
    """Soft-delete a document and verify delete response."""

    source_file = project_root / "Source_Code_Sample.py"
    document_id = await _upload_file(api_client, source_file)
    _ = await _wait_until_ready(api_client, document_id)

    delete_response = await api_client.delete(f"/documents/{document_id}")
    delete_response.raise_for_status()
    payload = delete_response.json()
    assert payload["document_id"] == document_id
    assert payload["deleted"] is True
    assert payload["hard_delete"] is False
