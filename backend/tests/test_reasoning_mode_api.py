from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.app.api.dependencies import create_runtime, get_runtime
from backend.app.database import CaseRepository, init_database
from backend.app.main import app
from backend.app.services.container import create_service_container


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def mode_client(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_ADVANCED", "0")
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    runtime = create_runtime(create_service_container(repository))
    calls = []

    class CapturingRuntime:
        @staticmethod
        def available(model_id):
            return True

        async def run(self, **kwargs):
            calls.append(kwargs)
            mode = kwargs["reasoning_mode"]
            return {"response": "Python可以读写CSV。", "tool_calls": [], "provider": "test",
                    "task_plan": [], "artifacts": [], "agent_trace": [{"reasoning_mode": mode}],
                    "proposed_action": None}

    runtime.harness.open_agent = CapturingRuntime()

    async def override_runtime():
        return runtime

    app.dependency_overrides[get_runtime] = override_runtime
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client, runtime, calls
            if runtime.tasks:
                await asyncio.gather(*tuple(runtime.tasks))
    finally:
        app.dependency_overrides.pop(get_runtime, None)
        repository.engine.dispose()


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", ["/v1/agent/runs", "/v1/agent/runs/dispatch"])
@pytest.mark.parametrize("mode", [None, "standard", "deep"])
async def test_mode_reaches_runtime_independently_of_answer_length(mode_client, endpoint, mode):
    client, runtime, calls = mode_client
    data = {"message": "解释Python读取CSV", "detail_level": "detailed", "conversation_id": "shared-conversation"}
    if mode is not None:
        data["reasoning_mode"] = mode
    response = await client.post(endpoint, data=data)
    assert response.status_code == 200, response.text
    if endpoint.endswith("dispatch"):
        await asyncio.gather(*tuple(runtime.tasks))
    assert calls[-1]["reasoning_mode"] == (mode or "standard")
    assert calls[-1]["conversation_id"] == "shared-conversation"
    saved = runtime.services.repository.get_agent_run(response.json()["run_id"])["state"]
    assert saved["reasoning_mode"] == (mode or "standard")
    assert saved["conversation_id"] == "shared-conversation"


@pytest.mark.anyio
async def test_same_conversation_can_switch_out_of_deep_mode(mode_client):
    client, _, calls = mode_client
    for mode in ("deep", "standard"):
        response = await client.post("/v1/agent/runs", data={
            "message": "解释Python读取CSV", "conversation_id": "same-chat", "reasoning_mode": mode,
        })
        assert response.status_code == 200, response.text
    assert [call["reasoning_mode"] for call in calls] == ["deep", "standard"]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", [None, "standard"])
async def test_resume_keeps_saved_mode_unless_explicitly_switched(mode_client, mode):
    client, runtime, calls = mode_client
    repository = runtime.services.repository
    repository.create_case("CASE_MODE", "/tmp/mode-test.png", "hash", "local", "clinician")
    run_id = runtime.harness._new_run(0, "CASE_MODE", "原任务", reasoning_mode="deep")
    runtime.harness._save_request_context(run_id, reasoning_mode="deep", conversation_id="resume-chat")
    repository.update_agent_run(run_id, status="PAUSED", current_node="paused")
    payload = {"message": "解释Python读取CSV", "detail_level": "concise"}
    if mode is not None:
        payload["reasoning_mode"] = mode
    response = await client.post(f"/v1/agent/runs/{run_id}/resume", json=payload)
    assert response.status_code == 200, response.text
    assert calls[-1]["reasoning_mode"] == (mode or "deep")
    assert calls[-1]["conversation_id"] == "resume-chat"
    assert repository.get_agent_run(run_id)["state"]["reasoning_mode"] == (mode or "deep")


@pytest.mark.anyio
async def test_invalid_mode_is_rejected_before_runtime(mode_client):
    client, _, calls = mode_client
    response = await client.post("/v1/agent/runs", data={
        "message": "解释Python", "reasoning_mode": "detailed",
    })
    assert response.status_code == 422
    assert calls == []
