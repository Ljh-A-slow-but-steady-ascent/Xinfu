import httpx
import pytest

from backend.app.main import app
from backend.app.services import knowledge_catalog


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_business_api_keeps_active_contracts_and_retires_diagnostic_views():
    paths = app.openapi()["paths"]
    for path in ("/v1/cases", "/v1/reports", "/v1/agent/runs/dispatch",
                 "/v1/agent/documents", "/v1/workspace/folders", "/v1/knowledge/overview",
                 "/v1/rare-disease/infer", "/v1/observatory/model-snapshot"):
        assert path in paths
    for path in ("/v1/audit", "/v1/cases/{case_id}/cloud-payload",
                 "/v1/cases/{case_id}/chat-history", "/v1/agent/runs/{run_id}/events",
                 "/v1/clinical-metadata/extract", "/v1/observatory/ham7/curves"):
        assert path not in paths


@pytest.mark.anyio
async def test_knowledge_and_observatory_resources_remain_readable():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        overview = await client.get("/v1/knowledge/overview")
        assert overview.status_code == 200
        assert overview.json()["domain_total_chunks"] == 1_082_000
        assert overview.json()["domain_count"] == 10
        items = await client.get("/v1/knowledge/items")
        assert items.status_code == 200
        document = await client.get(f'/v1/knowledge/items/{items.json()["items"][0]["document_id"]}')
        assert document.status_code == 200
        assert document.json()["chunks"]
        snapshot = await client.get("/v1/observatory/model-snapshot")
        assert snapshot.status_code == 200
        assert snapshot.json()["layers"]
        assert snapshot.json()["weight_histograms"]


@pytest.mark.anyio
async def test_knowledge_upload_is_queued_without_changing_index_statistics(monkeypatch, tmp_path):
    monkeypatch.setattr(knowledge_catalog, "UPLOAD_ROOT", tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        before = (await client.get("/v1/knowledge/overview")).json()
        response = await client.post("/v1/knowledge/uploads", files={"file": ("guide.md", b"# Clinical guide", "text/markdown")})
        assert response.status_code == 200
        assert response.json()["status"] == "PENDING_REVIEW"
        after = (await client.get("/v1/knowledge/overview")).json()
        assert after["pending_upload_count"] == before["pending_upload_count"] + 1
        assert after["chunk_count"] == before["chunk_count"]
        assert after["domain_total_chunks"] == before["domain_total_chunks"]
