"""Integration tests for retrieval query endpoint."""

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
    return response.json()["document_id"]


async def _wait_until_ready(client: AsyncClient, document_id: str, timeout_seconds: int = 180) -> None:
    """Poll status until ingestion is complete."""

    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/documents/{document_id}")
        response.raise_for_status()
        status = response.json()["status"]
        if status in {"ready", "partial", "failed"}:
            assert status in {"ready", "partial"}
            return
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for document {document_id}")


async def test_query_proxy_failure_returns_code_chunk(api_client: AsyncClient, project_root: Path) -> None:
    """Query should return Source_Code_Sample.py chunk mentioning report_failure."""

    source_file = project_root / "Source_Code_Sample.py"
    document_id = await _upload_file(api_client, source_file)
    await _wait_until_ready(api_client, document_id)

    query_response = await api_client.post(
        "/query",
        json={"query": "proxy failure handling", "top_k": 5, "filters": {"file_type": "py"}},
    )
    query_response.raise_for_status()
    payload = query_response.json()

    assert payload["results"]
    combined_content = "\n".join(result["content"] for result in payload["results"])
    assert "report_failure" in combined_content


async def test_query_ai_orchestration_returns_pdf_chunk(api_client: AsyncClient, project_root: Path) -> None:
    """Query should return non-empty results from Knowledge_Base_Sample.pdf."""

    pdf_file = project_root / "Knowledge_Base_Sample.pdf"
    document_id = await _upload_file(api_client, pdf_file)
    await _wait_until_ready(api_client, document_id)

    query_response = await api_client.post(
        "/query",
        json={"query": "AI orchestration", "top_k": 3, "filters": {"file_type": "pdf"}},
    )
    query_response.raise_for_status()
    payload = query_response.json()

    assert payload["results"]
    assert any(result["file_name"] == "Knowledge_Base_Sample.pdf" for result in payload["results"])
