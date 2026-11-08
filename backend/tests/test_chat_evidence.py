from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from backend.app.harness import open_source_agent as module
from backend.app.harness.open_source_agent import (
    EvidenceReadInput,
    EvidenceSearchInput,
    OpenSourceAgentRuntime,
    validate_chat_citations,
)
from backend.app.services.privacy import EgressGuard


CHUNK = {
    "id": "GUIDE_TEST#C01", "title": "测试指南", "content": "病灶变化需要进一步评估。",
    "citation": {"organization": "测试机构", "section": "评估"},
    "source": "https://example.org/guideline", "source_kind": "guideline", "authoritative": True,
}


class FakeRegistry:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.repository = SimpleNamespace(list_agent_documents=lambda **kwargs: [])

    async def invoke(self, name, arguments, context):
        self.calls.append((name, arguments))
        return SimpleNamespace(result={
            "chunks": [CHUNK],
            "evidence_assembly": {"evidence_sufficient": False, "assessment": "待Agent评估"},
        })


def test_chat_citation_validation_rejects_fabrication_without_claiming_entailment():
    checked, validation = validate_chat_citations(
        "建议进一步评估。[证据:GUIDE_TEST#C01] 已经确诊。[证据:MADE_UP]", [CHUNK]
    )
    assert "[证据:GUIDE_TEST#C01]" in checked
    assert "MADE_UP" not in checked
    assert "证据不足" in checked
    assert validation["valid_ids"] == ["GUIDE_TEST#C01"]
    assert validation["invalid_ids"] == ["MADE_UP"]
    assert validation["semantic_support_verified"] is False
    assert validation["citation_ids_valid"] is False


def test_valid_chat_reference_is_identity_check_only():
    checked, validation = validate_chat_citations("需复核。[证据：GUIDE_TEST#C01]", [CHUNK])
    assert checked == "需复核。[证据:GUIDE_TEST#C01]"
    assert validation["citation_ids_valid"] is True
    assert validation["semantic_support_verified"] is False


def test_local_reference_is_valid_identity_but_not_medical_evidence():
    local_rule = {**CHUNK, "source_kind": "local_rule", "source": "local_triage_guide"}
    checked, validation = validate_chat_citations("需复核。[证据:GUIDE_TEST#C01]", [local_rule])
    assert validation["citation_ids_valid"] is True
    assert validation["medical_source_ids"] == []
    assert validation["local_ids"] == [CHUNK["id"]]
    assert "不能作为医学指南依据" in checked


def test_evidence_tool_schemas_bound_context_size():
    with pytest.raises(ValidationError):
        EvidenceSearchInput(queries=["a"] * 4)
    with pytest.raises(ValidationError):
        EvidenceReadInput(evidence_ids=["a"] * 6)
    with pytest.raises(ValidationError):
        EvidenceSearchInput(queries=["x" * 501])


@pytest.mark.anyio
async def test_shared_evidence_budget_and_duplicate_query_cache(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_MAX_TOOL_CALLS", "3")
    registry = FakeRegistry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    state: dict = {}
    tools = {tool.name: tool for tool in runtime._tools(
        7, "conversation", None, lambda *args: None, [], rag_state=state,
    )}
    first = json.loads(await tools["search_evidence"].ainvoke({"queries": ["黑色素瘤"]}))
    repeated = json.loads(await tools["search_evidence"].ainvoke({"queries": ["黑色素瘤"]}))
    read = json.loads(await tools["read_evidence"].ainvoke({"evidence_ids": [CHUNK["id"]]}))
    stopped = json.loads(await tools["search_evidence"].ainvoke({"queries": ["转诊条件"]}))
    assert first["chunks"] == read["chunks"]
    assert repeated["cache_hit"] is True
    assert len(registry.calls) == 2
    assert state["tool_calls"] == 3
    assert stopped["budget_exhausted"] is True
    assert "停止调用" in stopped["next_action"]
    assert list(state["chunks"]) == [CHUNK["id"]]
    assert [entry["tool"] for entry in state["trace"]] == ["search_evidence", "search_evidence", "read_evidence"]
    assert state["trace"][1]["status"] == "cached"


@pytest.mark.anyio
async def test_unknown_read_consumes_attempt_and_cannot_access_arbitrary_ids(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_MAX_TOOL_CALLS", "2")
    registry = FakeRegistry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    state: dict = {}
    tools = {tool.name: tool for tool in runtime._tools(
        7, "conversation", None, lambda *args: None, [], rag_state=state,
    )}
    rejected = json.loads(await tools["read_evidence"].ainvoke({"evidence_ids": [CHUNK["id"]]}))
    assert rejected["chunks"] == []
    assert rejected["unknown_evidence_ids"] == [CHUNK["id"]]
    assert registry.calls == []
    await tools["search_evidence"].ainvoke({"queries": ["黑色素瘤"]})
    stopped = json.loads(await tools["read_evidence"].ainvoke({"evidence_ids": [CHUNK["id"]]}))
    assert stopped["budget_exhausted"] is True
    assert len(registry.calls) == 1
    assert state["trace"][0]["status"] == "rejected"


@pytest.mark.anyio
async def test_legacy_search_is_local_tool_alias_without_nested_agent(monkeypatch):
    monkeypatch.delenv("DERMAGUARD_RAG_MAX_TOOL_CALLS", raising=False)
    monkeypatch.setattr(module, "get_config", lambda: SimpleNamespace(rag=SimpleNamespace(max_tool_calls=1)))
    registry = FakeRegistry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    tools = {tool.name: tool for tool in runtime._tools(7, "conversation", None, lambda *args: None, [])}
    await tools["search_guideline"].ainvoke({"query": "转诊条件"})
    assert registry.calls == [("search_evidence", {"queries": ["转诊条件"], "top_k": 5})]
    stopped = json.loads(await tools["search_evidence"].ainvoke({"queries": ["病灶变化"]}))
    assert stopped["budget_exhausted"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("with_citation", [True, False])
async def test_chat_keeps_native_tool_loop_and_reports_actual_reference_status(monkeypatch, with_citation):
    registry = FakeRegistry()
    registry.repository.patient_identifiers_for_user = lambda user_id: []
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    monkeypatch.setattr(runtime, "_model", lambda model_id: SimpleNamespace(model_name="fake"))

    class FakeAgent:
        def __init__(self, tools):
            self.tools = {tool.name: tool for tool in tools}

        async def aget_state(self, config):
            return SimpleNamespace(values={})

        async def astream(self, inputs, config, stream_mode):
            # The model chooses different queries after an initial observation, then reads.
            for index, (name, arguments) in enumerate([
                ("search_evidence", {"queries": ["病灶变化"]}),
                ("search_evidence", {"queries": ["转诊条件"]}),
                ("read_evidence", {"evidence_ids": [CHUNK["id"]]}),
            ]):
                result = await self.tools[name].ainvoke(arguments)
                yield {"tools": {"messages": [ToolMessage(content=result, tool_call_id=str(index), name=name)]}}
            answer = json.dumps({"claims": [{"text": "建议进一步评估。", "citations": [{
                "evidence_id": CHUNK["id"], "quote": CHUNK["content"]}]}],
                "sufficient": True}) if with_citation else "建议进一步评估。"
            yield {"model": {"messages": [AIMessage(content=answer)]}}

    monkeypatch.setattr(module, "create_deep_agent", lambda **kwargs: FakeAgent(kwargs["tools"]))
    result = await runtime.run(
        message="病灶为什么要转诊？", model_id="auto", conversation_id="conversation", user_id=7,
        case_id=None, history=[], on_event=lambda *args: None, reasoning_mode="deep",
    )
    assert [name for name, _ in registry.calls] == ["search_evidence", "search_evidence", "read_evidence"]
    assert len(result["retrieval_trace"]) == 3
    trace = result["agent_trace"][0]
    assert trace["evidence_sufficient"] is with_citation
    assert trace["semantic_support_verified"] is False
    assert trace["retrieval_query"] == [["病灶变化"], ["转诊条件"]]
    if not with_citation:
        assert "尚未获得可核对的本地医学证据引用" in result["response"]
