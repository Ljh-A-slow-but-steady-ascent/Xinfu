from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
import threading

from anyio import to_thread
import httpx
import pytest
from sqlalchemy.pool import StaticPool

from backend.app.api import rare_disease_routes
from backend.app.api.dependencies import create_runtime, get_runtime
from backend.app.care.schemas import ClinicalMetadata
from backend.app.database import CaseRepository, init_database
from backend.app.main import app
from backend.app.services.container import create_service_container


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeVision:
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        assert image_path.is_file()
        return {
            "model_name": "fake-local-vision", "latency_ms": 1.0,
            "top_predictions": [{"label": "nevus", "score": 0.8}, {"label": "melanoma", "score": 0.2}],
        }


async def _wait_started(started: threading.Event) -> None:
    async def wait():
        while not started.is_set():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), timeout=1.0)


@pytest.mark.anyio
async def test_case_analysis_keeps_health_responsive_and_persists_worker_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_ADVANCED", "0")
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    services = create_service_container(repository)
    services.cases.data_root = tmp_path / "cases"
    services.visual.runner = FakeVision()
    patient = repository.create_patient(
        name="并发测试", sex="男", age=65, outpatient_no="ASYNC-001", phone=None,
        medical_history="", allergy_history="", created_by=0,
    )
    image = Path("third_party/DermFM-Zero/examples/PAT_8_15_820.png").read_bytes()
    case_id = services.cases.create(image, "image/png", mode="clinician", retention_mode="session")
    metadata = ClinicalMetadata(
        age=65, gender="MALE", region="FOREARM", bleed=True, itch=False, hurt=False,
        grew=True, changed=True, diameter_1=9, diameter_2=7, skin_cancer_history=False, cancer_history=False,
    )
    repository.link_case_patient(case_id, patient.patient_id, "左前臂", "近期变化并出血", metadata.model_dump(mode="json"))
    runtime = create_runtime(services)
    original = services.analysis.run
    started, release = threading.Event(), threading.Event()
    request_thread = threading.get_ident()

    def slow_analysis(value):
        assert threading.get_ident() != request_thread
        started.set()
        assert release.wait(3), "test did not release the blocking analysis"
        return original(value)

    monkeypatch.setattr(services.analysis, "run", slow_analysis)

    async def override_runtime():
        return runtime

    app.dependency_overrides[get_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post(f"/v1/cases/{case_id}/analyze"))
            try:
                await _wait_started(started)
                health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                assert health.status_code == 200
                assert not pending.done(), "long analysis must still be pending while health responds"
            finally:
                release.set()
            response = await asyncio.wait_for(pending, timeout=3)
            assert response.status_code == 200, response.text
            assert response.json()["rag"]["retrieval"]["mode"] == "retrieval_only"
            assert repository.get_visual(case_id)["top_predictions"][0]["label"] == "nevus"
            assert repository.get_care_pathway(case_id)
    finally:
        release.set()
        app.dependency_overrides.pop(get_runtime, None)
        repository.engine.dispose()


@pytest.mark.anyio
async def test_rare_rag_endpoint_does_not_block_other_requests(monkeypatch):
    started, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()

    def slow_rare(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        started.set()
        assert release.wait(3), "test did not release the blocking rare-disease request"
        return {"package_id": "RARE_TEST", "answer": "仅供测试", "evidence_sufficient": False}

    monkeypatch.setattr(rare_disease_routes.rare, "infer", slow_rare)
    data = {"mode": "text_only", "text": "皮肤病灶"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        pending = asyncio.create_task(client.post("/v1/rare-disease/infer", data=data))
        try:
            await _wait_started(started)
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            assert health.json() == {"status": "ok"}
            assert not pending.done()
        finally:
            release.set()
        response = await asyncio.wait_for(pending, timeout=3)
        assert response.status_code == 200, response.text
        assert response.json()["package_id"] == "RARE_TEST"


@pytest.mark.anyio
async def test_in_memory_database_is_shared_across_worker_threads():
    engine = init_database("sqlite+pysqlite:///:memory:")
    assert isinstance(engine.pool, StaticPool)
    repository = CaseRepository(engine)
    try:
        created = await to_thread.run_sync(partial(
            repository.create_patient, name="跨线程测试", sex="女", age=42,
            outpatient_no="THREAD-001", phone=None, medical_history="", allergy_history="", created_by=0,
        ))
        assert repository.get_patient(created.patient_id).name == "跨线程测试"
        patient = await to_thread.run_sync(repository.get_patient, created.patient_id)
        assert patient.outpatient_no == "THREAD-001"
    finally:
        engine.dispose()


def test_file_database_keeps_normal_pool(tmp_path):
    engine = init_database(f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    try:
        assert not isinstance(engine.pool, StaticPool)
    finally:
        engine.dispose()
