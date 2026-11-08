from __future__ import annotations

import json

import pytest
import requests

from backend.app.services.privacy import EgressGuard
from backend.app.services.rag.llm import RagLLM, RagLLMError


TOOLS = [{"type": "function", "function": {
    "name": "search_evidence", "description": "Search local evidence",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}}]


class Response:
    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": self.message}]}


@pytest.mark.parametrize("provider,key,model,base", [
    ("deepseek", "DEEPSEEK_API_KEY", "deepseek-flash", "https://api.deepseek.com/chat/completions"),
    ("zhipu", "ZHIPUAI_API_KEY", "glm-5.3-flash", "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
    ("groq", "GROQ_API_KEY", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1/chat/completions"),
])
def test_native_tools_use_configured_provider(monkeypatch, provider, key, model, base):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", provider)
    monkeypatch.setenv(key, "test-secret")
    for name in ("DEEPSEEK_MODEL", "ZHIPUAI_MODEL", "GROQ_MODEL", "DERMAGUARD_RAG_MODEL",
                 "DEEPSEEK_BASE_URL", "ZHIPUAI_BASE_URL", "GROQ_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    captured = {}
    call = {"id": "call_1", "type": "function", "function": {"name": "search_evidence", "arguments": '{"query":"黑色素瘤"}'}}

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response({"role": "assistant", "content": None, "tool_calls": [call]})

    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", post)
    client = RagLLM()
    result = client.step([{"role": "user", "content": "黑色素瘤"}], TOOLS, timeout=2.5)
    assert client.available
    assert captured["url"] == base
    assert captured["json"]["model"] == model
    assert captured["json"]["tools"] == TOOLS
    assert captured["json"]["tool_choice"] == "auto"
    assert "response_format" not in captured["json"]
    assert captured["timeout"] == 2.5
    assert captured["json"]["max_tokens"] <= 1600
    assert result["tool_calls"] == [call]
    assert isinstance(result["tool_calls"][0]["function"]["arguments"], str)


def test_mock_never_networks_even_with_explicit_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", lambda *a, **k: pytest.fail("offline request"))
    client = RagLLM(model_id="deepseek-flash")
    assert not client.available
    with pytest.raises(RagLLMError, match="离线"):
        client.step([{"role": "user", "content": "hello"}], TOOLS)


def test_explicit_model_selects_its_credentials_and_normalizes_endpoint(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "groq")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/api/")
    captured = {}

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response({"content": "需要本地证据。"})

    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", post)
    RagLLM(model_id="deepseek-v4-pro").step([{"role": "user", "content": "hello"}], [])
    assert captured["url"] == "https://example.invalid/api/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer deepseek-key"
    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert "tool_choice" not in captured["json"]


def test_missing_credentials_is_unavailable(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = RagLLM(model_id="deepseek-flash")
    assert not client.available


def test_all_text_arguments_and_tool_results_are_sanitized(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return Response({"content": "不足，需补充证据。"})

    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", post)
    messages = [
        {"role": "user", "content": "电话13800138000，身份证110101199001011234"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "search_evidence", "arguments": json.dumps({"query": "电话13800138000"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"excerpt": "电话13800138000"})},
    ]
    original = json.dumps(messages)
    client = RagLLM(EgressGuard())
    client.step(messages, TOOLS)
    wire = json.dumps(captured["json"]["messages"])
    assert "13800138000" not in wire
    assert "110101199001011234" not in wire
    assert "REDACTED_PHONE" in wire
    assert captured["json"]["messages"][1]["tool_calls"][0]["function"]["name"] == "search_evidence"
    assert json.dumps(messages) == original


@pytest.mark.parametrize("message", [
    {"role": "user", "content": "data:image/png;base64,iVBORw0KGgoAAA"},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"image_path": "/tmp/private.png"})},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "search_evidence", "arguments": json.dumps({"name": "patient"})}}]},
])
def test_prohibited_content_is_blocked_before_network(monkeypatch, message):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", lambda *a, **k: pytest.fail("blocked request"))
    with pytest.raises(RagLLMError, match="出站检查"):
        RagLLM().step([message], TOOLS)


def test_timeout_is_single_attempt_and_safe(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-secret")
    monkeypatch.setenv("DERMAGUARD_RAG_LLM_TIMEOUT", "4")
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["timeout"])
        raise requests.Timeout("private-secret raw provider message")

    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", post)
    client = RagLLM()
    with pytest.raises(RagLLMError, match="超时") as caught:
        client.step([{"role": "user", "content": "hello"}], TOOLS)
    assert calls == [4.0]
    assert "private-secret" not in str(caught.value)
    assert client.last_error == "RAG 模型响应超时"


@pytest.mark.parametrize("configured,remaining,expected", [
    ("12", 45.0, 12.0), ("12", 2.0, 2.0), ("60", 45.0, 30.0),
])
def test_remaining_budget_can_only_reduce_request_timeout(monkeypatch, configured, remaining, expected):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DERMAGUARD_RAG_LLM_TIMEOUT", configured)
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return Response({"content": "需要证据。"})

    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", post)
    RagLLM().step([{"role": "user", "content": "hello"}], TOOLS, timeout=remaining)
    assert captured["timeout"] == expected


@pytest.mark.parametrize("content", [None, [], {"tool_calls": "invalid"}])
def test_malformed_model_responses_fail_closed(monkeypatch, content):
    monkeypatch.setenv("DERMAGUARD_CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr("backend.app.services.rag.llm.requests.post", lambda *a, **k: Response(content))
    with pytest.raises(RagLLMError, match="响应无效"):
        RagLLM().step([{"role": "user", "content": "hello"}], TOOLS)
