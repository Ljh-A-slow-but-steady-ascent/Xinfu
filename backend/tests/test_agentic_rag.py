from __future__ import annotations

import json

import pytest

from backend.app.config import AppConfig, RagConfig
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.rag.agent import AgenticRag
from backend.app.services.rag.evidence import EvidenceAnswer, EvidenceSession, validate_answer
from backend.app.services.rag.retriever import LocalRagRetriever


def tool(name, **arguments):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": f"call_{name}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def final_from_observation(messages, *, sufficient=True, gaps=None):
    chunks = json.loads(next(item["content"] for item in reversed(messages) if item["role"] == "tool"))["chunks"]
    chunk = next(item for item in chunks if item["authoritative"])
    return {"role": "assistant", "content": json.dumps({
        "claims": [{"text": "需要结合医生评估及给定证据核对。", "citations": [{
            "evidence_id": chunk["id"], "quote": chunk["content"][:30]}]}],
        "sufficient": sufficient, "missing_information": gaps or []}, ensure_ascii=False)}


class ScriptedModel:
    available = True
    model = "scripted-tool-model"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.calls = []

    def step(self, messages, tools, timeout=None):
        self.calls.append({"messages": list(messages), "tools": tools, "timeout": timeout})
        result = next(self.steps)
        return result(messages) if callable(result) else result


@pytest.fixture
def retriever(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_ADVANCED", "0")
    return LocalRagRetriever()


def test_agent_rewrites_search_reads_and_finishes_with_checked_quote(retriever):
    def read_chosen(messages):
        chunk = json.loads(messages[-1]["content"])["chunks"][0]
        return tool("read_evidence", evidence_ids=[chunk["id"]], include_neighbors=True)
    model = ScriptedModel([
        tool("search_evidence", queries=["basal cell carcinoma 出血 活检"], top_k=3),
        read_chosen, final_from_observation,
    ])
    result = AgenticRag(retriever, model).run("这个病灶需要什么检查？")
    assert result["evidence_sufficient"] is True
    assert result["citation_validation"]["quote_spans_verified"] is True
    assert result["retrieval"]["model_calls"] == 3
    assert [entry["tool"] for entry in result["agent_trace"]] == ["search_evidence", "read_evidence"]
    assert result["agent_trace"][0]["arguments"]["queries"] == ["basal cell carcinoma 出血 活检"]
    assert "这个病灶需要什么检查" in model.calls[0]["messages"][1]["content"]
    assert result["chunks"][0]["verified_citation"]


def test_agent_can_choose_more_than_two_search_rounds(retriever):
    model = ScriptedModel([
        tool("search_evidence", queries=["黑色素瘤 危险征象"]),
        tool("search_evidence", queries=["黑色素瘤 活检 确认"]),
        tool("search_evidence", queries=["黑色素瘤 转诊 NICE"]),
        final_from_observation,
    ])
    result = AgenticRag(retriever, model).run("如何鉴别和转诊？")
    assert len(result["agent_trace"]) == 3
    assert result["retrieval"]["stop_reason"] == "completed"


def test_simple_question_has_no_mandatory_planner_or_judge_calls(retriever):
    result = AgenticRag(retriever, ScriptedModel([
        tool("search_evidence", queries=["melanoma"]), final_from_observation,
    ])).run("黑色素瘤的警示表现")
    assert result["retrieval"]["model_calls"] == 2
    assert result["retrieval"]["tool_calls"] == 1


def test_no_answer_is_not_upgraded_by_high_retrieval_score(retriever):
    result = AgenticRag(retriever, ScriptedModel([
        tool("search_evidence", queries=["北京天气变化怎么预测"]),
        {"role": "assistant", "content": json.dumps({"claims": [], "sufficient": False,
            "missing_information": ["本地医学知识库不包含天气预测"]}, ensure_ascii=False)},
    ])).run("北京天气变化怎么预测")
    assert result["evidence_sufficient"] is False
    assert result["claims"] == []


def test_unknown_or_invented_quotes_are_removed_and_never_sufficient(retriever):
    model = ScriptedModel([
        tool("search_evidence", queries=["melanoma"]),
        {"role": "assistant", "content": json.dumps({"claims": [{"text": "未经支持的结论", "citations": [
            {"evidence_id": "FAKE", "quote": "这是一段凭空编造的原文"}]}], "sufficient": True})},
    ])
    result = AgenticRag(retriever, model).run("黑色素瘤")
    assert not result["evidence_sufficient"]
    assert "未经支持的结论" not in result["answer"]
    assert result["citation_validation"]["errors"]


def test_model_can_repair_bad_citation_without_fixed_pipeline(retriever):
    retriever.config = AppConfig(rag=RagConfig(max_tool_calls=1, max_model_turns=4))
    state = {}
    def wrong(messages):
        state["chunks"] = json.loads(messages[-1]["content"])["chunks"]
        return {"role": "assistant", "content": json.dumps({"claims": [{"text": "有限结论", "citations": [
            {"evidence_id": "FAKE", "quote": "不能引用不曾返回的证据"}]}], "sufficient": True})}
    def repaired(messages):
        chunk = state["chunks"][0]
        return {"role": "assistant", "content": json.dumps({"claims": [{"text": "有限结论", "citations": [
            {"evidence_id": chunk["id"], "quote": chunk["content"][:30]}]}], "sufficient": True})}
    result = AgenticRag(retriever, ScriptedModel([
        tool("search_evidence", queries=["melanoma"]), wrong, repaired,
    ])).run("黑色素瘤")
    assert result["evidence_sufficient"]
    assert result["retrieval"]["model_calls"] == 3


def test_local_and_unseen_text_cannot_support_medical_answer():
    for observed in ({}, {"D": {"content": "这是本地规则规则，不是权威医学指南。", "authoritative": False}}):
        answer = EvidenceAnswer.model_validate({"claims": [{"text": "未经支持", "citations": [{
            "evidence_id": "D", "quote": "这是本地规则规则，不是权威医学指南。"}]}], "sufficient": True})
        assert not validate_answer(answer, observed)["evidence_sufficient"]


def test_offline_and_provider_failure_are_explicit(retriever):
    offline = KnowledgeService(retriever).retrieve("北京天气变化怎么预测")
    assert offline["retrieval"]["mode"] == "retrieval_only"
    assert offline["retrieval"]["model_calls"] == 0
    assert not offline["evidence_sufficient"]
    assert not offline["evidence_assembly"]["evidence_sufficient"]
    assert not KnowledgeService(retriever).references(offline["chunks"])
    failed = AgenticRag(retriever, ScriptedModel([])).run("黑色素瘤")
    assert failed["retrieval"]["stop_reason"] == "model_error"
    assert not failed["evidence_sufficient"]


def test_tool_attempt_budget_includes_invalid_calls(retriever):
    retriever.config = AppConfig(rag=RagConfig(max_tool_calls=1, max_model_turns=4))
    model = ScriptedModel([
        tool("unknown_tool", query="a"),
        tool("search_evidence", queries=["melanoma"]),
        {"role": "assistant", "content": '{"sufficient":false,"missing_information":["预算不足"]}'},
    ])
    result = AgenticRag(retriever, model).run("黑色素瘤")
    assert model.calls[1]["tools"] == []
    assert result["agent_trace"] == []
    assert not result["evidence_sufficient"]
    assert result["retrieval"]["tool_attempts"] == 2


def test_request_cache_does_not_share_observations(retriever):
    session = EvidenceSession(retriever)
    params = {"queries": ["melanoma"], "top_k": 3}
    first = session.invoke("search_evidence", params)
    second = session.invoke("search_evidence", params)
    assert first == second
    assert session.trace[-1]["cached"]
    fresh = EvidenceSession(retriever)
    with pytest.raises(ValueError, match="本次搜索"):
        fresh.invoke("read_evidence", {"evidence_ids": [first["chunks"][0]["id"]]})


def test_corpus_quote_outside_visible_preview_is_not_accepted():
    answer = EvidenceAnswer.model_validate({"claims": [{"text": "结论", "citations": [{
        "evidence_id": "A", "quote": "这段话没有出现在模型见过的预览里"}]}], "sufficient": True})
    result = validate_answer(answer, {"A": {"content": "模型只能看到这个摘要。", "authoritative": True}})
    assert not result["evidence_sufficient"]


def test_short_definition_can_be_quoted_without_padding():
    answer = EvidenceAnswer.model_validate({"claims": [{"text": "A表示不对称。", "citations": [{
        "evidence_id": "A", "quote": "A 不对称"}]}], "sufficient": True})
    result = validate_answer(answer, {"A": {"content": "ABCDE包括A 不对称；B 边界不规则。", "authoritative": True}})
    assert result["evidence_sufficient"]


def test_expired_deadline_skips_fallback_retrieval(retriever, monkeypatch):
    import backend.app.services.rag.agent as module
    moments = iter([0, 50, 50, 50])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    model = ScriptedModel([])
    result = AgenticRag(retriever, model).run("黑色素瘤")
    assert result["agent_trace"] == []
    assert result["retrieval"]["stop_reason"] == "time_budget"
    assert result["retrieval"]["model_calls"] == 0
    assert result["retrieval"]["tool_calls"] == 0
