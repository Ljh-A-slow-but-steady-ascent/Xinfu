from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from backend.app.harness import open_source_agent as module
from backend.app.harness.open_source_agent import OpenSourceAgentRuntime, agent_system_prompt
from backend.app.services.privacy import EgressGuard


CHUNK = {"id": "MEDICAL_1", "content": "病灶变化时可请医生进一步评估。", "title": "本地医学摘要",
         "source": "https://example.org/medical", "authoritative": True,
         "citation": {"organization": "测试来源", "section": "评估"}}


class Registry:
    def __init__(self):
        self.repository = SimpleNamespace(patient_identifiers_for_user=lambda user_id: [],
                                         list_agent_documents=lambda **kwargs: [])
        self.calls = []

    async def invoke(self, name, arguments, context):
        self.calls.append((name, arguments))
        return SimpleNamespace(result={"chunks": [CHUNK], "evidence_assembly": {"evidence_sufficient": False}})


def fake_runtime(monkeypatch, *, response: str, search: bool = False):
    registry = Registry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    monkeypatch.setattr(runtime, "_model", lambda model_id: SimpleNamespace(model_name="fake"))
    created_prompts = []

    class FakeAgent:
        def __init__(self, tools):
            self.tools = {tool.name: tool for tool in tools}

        async def aget_state(self, config):
            return SimpleNamespace(values={})

        async def astream(self, inputs, config, stream_mode):
            if search:
                result = await self.tools["search_evidence"].ainvoke({"queries": ["测试查询"]})
                yield {"tools": {"messages": [ToolMessage(content=result, name="search_evidence", tool_call_id="one")]}}
            yield {"model": {"messages": [AIMessage(content=response)]}}

    def create(**kwargs):
        created_prompts.append(kwargs["system_prompt"])
        return FakeAgent(kwargs["tools"])

    monkeypatch.setattr(module, "create_deep_agent", create)
    return runtime, registry, created_prompts


async def ask(runtime, question, mode="standard"):
    return await runtime.run(message=question, model_id="auto", conversation_id="same-conversation",
        user_id=1, case_id=None, history=[], on_event=lambda *args: None, reasoning_mode=mode)


@pytest.mark.anyio
@pytest.mark.parametrize("question,answer", [
    ("为什么冬天天气冷？", "冬天太阳高度较低，白昼较短，地面获得的太阳能较少。"),
    ("Python怎么对列表排序？", "可以用 `sorted(values)` 返回排序后的新列表。"),
    ("帮我写一句周末问候", "周末愉快，愿你有一段轻松自在的时光。"),
])
async def test_standard_general_answers_do_not_require_medical_evidence(monkeypatch, question, answer):
    runtime, registry, prompts = fake_runtime(monkeypatch, response=answer)
    result = await ask(runtime, question)
    assert result["response"] == answer
    assert registry.calls == []
    assert result["reasoning_mode"] == "standard"
    assert result["evidence_policy"] == "general_assistant"
    assert result["agent_trace"][0]["evidence_sufficient"] is None
    assert "[当前回答模式:standard]" in prompts[0]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["standard", "deep"])
async def test_incidental_medical_search_does_not_overwrite_general_answer(monkeypatch, mode):
    answer = "冷空气南下会带来降温，具体今天的温度需要实时天气来源。"
    runtime, _, _ = fake_runtime(monkeypatch, response=answer, search=True)
    result = await ask(runtime, "天气为什么突然变冷？", mode)
    assert result["response"].startswith(answer)
    assert "证据不足" not in result["response"]
    assert "未提供可逐条核对" not in result["response"]
    assert result["agent_trace"][0]["evidence_sufficient"] is None


@pytest.mark.anyio
async def test_standard_medical_markdown_remains_available_without_forced_json(monkeypatch):
    answer = "这类变化可以请医生复核。[证据:MEDICAL_1]"
    runtime, _, _ = fake_runtime(monkeypatch, response=answer, search=True)
    result = await ask(runtime, "病灶变化应如何评估？")
    assert result["response"].startswith(answer)
    assert "模型未提供" not in result["response"]
    assert result["citation_validation"]["citation_ids_valid"]
    assert result["agent_trace"][0]["evidence_sufficient"] is None


@pytest.mark.anyio
async def test_deep_medical_answer_uses_quote_validation(monkeypatch):
    raw = json.dumps({"claims": [{"text": "请医生进一步评估。", "citations": [{
        "evidence_id": CHUNK["id"], "quote": CHUNK["content"]}]}], "sufficient": True})
    runtime, _, prompts = fake_runtime(monkeypatch, response=raw, search=True)
    result = await ask(runtime, "病灶变化应如何评估？", "deep")
    assert result["response"].startswith("请医生进一步评估。 [证据:MEDICAL_1]")
    assert result["citation_validation"]["quote_spans_verified"]
    assert result["agent_trace"][0]["evidence_sufficient"] is True
    assert "[当前回答模式:deep]" in prompts[0]


@pytest.mark.anyio
async def test_deep_medical_fake_quote_is_not_presented_as_supported(monkeypatch):
    raw = json.dumps({"claims": [{"text": "已经确诊。", "citations": [{
        "evidence_id": CHUNK["id"], "quote": "这句话没有出现在本地证据里"}]}], "sufficient": True})
    runtime, _, _ = fake_runtime(monkeypatch, response=raw, search=True)
    result = await ask(runtime, "病灶变化应如何评估？", "deep")
    assert "已经确诊。" not in result["response"]
    assert result["agent_trace"][0]["evidence_sufficient"] is False


@pytest.mark.anyio
async def test_deep_medical_prose_without_quotes_is_filtered_after_medical_search(monkeypatch):
    runtime, _, _ = fake_runtime(monkeypatch, response="已经确诊，必须立刻用药。", search=True)
    result = await ask(runtime, "病灶变化应如何评估？", "deep")
    assert "已经确诊" not in result["response"]
    assert "未提供可逐条核对" in result["response"]
    assert result["agent_trace"][0]["evidence_sufficient"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["standard", "deep"])
async def test_programming_json_claims_field_is_not_a_medical_protocol(monkeypatch, mode):
    answer = '{"claims":["example"],"status":"ok"}'
    runtime, _, _ = fake_runtime(monkeypatch, response=answer)
    result = await ask(runtime, "请给我一个含claims字段的JSON例子", mode)
    assert result["response"] == answer
    assert result["agent_trace"][0]["evidence_sufficient"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("search", [False, True])
async def test_deep_medical_vocabulary_inside_export_task_still_returns_code(monkeypatch, search):
    answer = "可以用 csv.DictWriter 导出记录：\n```python\nwriter.writerows(records)\n```"
    runtime, _, _ = fake_runtime(monkeypatch, response=answer, search=search)
    result = await ask(runtime, "用Python导出皮肤检查记录CSV", "deep")
    assert result["response"] == answer
    assert "医学证据" not in result["response"]
    assert result["agent_trace"][0]["evidence_sufficient"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["standard", "deep"])
@pytest.mark.parametrize("search", [False, True])
async def test_general_code_urls_and_markdown_links_are_preserved(monkeypatch, mode, search):
    answer = ("```python\nimport requests\nrequests.get('https://example.com/api?q=1')\n```\n"
              "[Python文档](https://docs.python.org/3/)")
    runtime, _, _ = fake_runtime(monkeypatch, response=answer, search=search)
    result = await ask(runtime, "给一个requests.get示例并附Python文档链接", mode)
    assert result["response"] == answer
    assert "未验证链接" not in result["response"]


@pytest.mark.anyio
async def test_grounded_medical_explanation_still_filters_unverified_source_urls(monkeypatch):
    raw = json.dumps({"claims": [{"text": "需医生评估，参见 https://unverified.example.com/guide", "citations": [{
        "evidence_id": CHUNK["id"], "quote": CHUNK["content"]}]}], "sufficient": True})
    runtime, _, _ = fake_runtime(monkeypatch, response=raw, search=True)
    result = await ask(runtime, "病灶变化应如何评估？", "deep")
    assert "unverified.example.com" not in result["response"]
    assert "[未验证链接已移除]" in result["response"]


@pytest.mark.anyio
async def test_deep_nonmedical_still_solves_programming(monkeypatch):
    answer = "用字典记录已见元素，可以在线性时间内查找重复项。"
    runtime, registry, _ = fake_runtime(monkeypatch, response=answer)
    result = await ask(runtime, "深入分析如何高效查找列表重复项", "deep")
    assert result["response"] == answer
    assert registry.calls == []
    assert result["agent_trace"][0]["evidence_sufficient"] is None


class RecordingModel(BaseChatModel):
    model_name: str = "mode-policy-test"
    seen_messages: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "recording-mode-policy"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="我会继续处理当前问题。"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)


@pytest.mark.anyio
async def test_real_deep_agent_mode_switch_reuses_history_without_stale_policy(monkeypatch):
    registry = Registry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    model = RecordingModel()
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    for question, mode in [("记住数字七", "standard"), ("深入解释这个数字", "deep"), ("刚才的数字是什么", "standard")]:
        await ask(runtime, question, mode)
    assert len(model.seen_messages) == 3
    for messages, mode in zip(model.seen_messages, ["standard", "deep", "standard"]):
        systems = [item for item in messages if isinstance(item, SystemMessage)]
        assert len(systems) == 1
        content = systems[0].content
        content = content if isinstance(content, str) else "\n".join(item.get("text", "") for item in content)
        assert f"[当前回答模式:{mode}]" in content
        other = "deep" if mode == "standard" else "standard"
        assert f"[当前回答模式:{other}]" not in content
    final_user_text = [item.content for item in model.seen_messages[-1] if isinstance(item, HumanMessage)]
    assert final_user_text == ["记住数字七", "深入解释这个数字", "刚才的数字是什么"]


def test_mode_policies_offer_general_search_and_reading():
    for mode in ("standard", "deep"):
        prompt = agent_system_prompt(mode)
        assert "web_search" in prompt
        assert "read_webpage" in prompt
        assert "自主组织和改写查询" in prompt
        assert "Asia/Shanghai" in prompt
        assert "非医学" in prompt


class PublicWebModel(RecordingModel):
    query: str
    source_url: str
    answer: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name == "read_webpage":
            reply = AIMessage(content=self.answer)
        elif isinstance(last, ToolMessage) and last.name == "web_search":
            source = json.loads(last.content)["results"][0]["url"]
            reply = AIMessage(content="", tool_calls=[{"name": "read_webpage", "args": {"url": source}, "id": "read-one"}])
        else:
            reply = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": self.query}, "id": "search-one"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.mark.anyio
@pytest.mark.parametrize("question,query,url,answer", [
    ("联网查Python read_text参数", "pathlib read_text site:docs.python.org", "https://docs.python.org/3/library/pathlib.html", "read_text支持encoding、errors和newline参数。"),
    ("查国博开放时间", "中国国家博物馆 开放时间 官网", "https://www.chnmuseum.cn/cg/", "开放时间为9:00至17:00。"),
])
async def test_real_agent_general_search_read_answer_loop(monkeypatch, question, query, url, answer):
    class PublicRegistry(Registry):
        async def invoke(self, name, arguments, context):
            self.calls.append((name, arguments))
            assert name in {"web_search", "read_webpage"}
            sources = [{"title": "官方资料", "url": url}]
            if name == "web_search":
                sources.append({"title": "未阅读的搜索结果", "url": "https://unused.example.org/"})
            return SimpleNamespace(result={"status": "ok", "content": answer,
                "results": sources})

    registry = PublicRegistry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    model = PublicWebModel(query=query, source_url=url, answer=answer)
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    result = await ask(runtime, question)
    assert [name for name, _ in registry.calls] == ["web_search", "read_webpage"]
    assert registry.calls[0][1]["query"] == query
    assert registry.calls[1][1]["url"] == url
    assert result["response"].startswith(answer)
    assert url in result["response"]
    assert "unused.example.org" not in result["response"]
    assert result["agent_trace"][0]["evidence_sufficient"] is None


class WeatherModel(RecordingModel):
    has_city: bool = True
    available_tools: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.available_tools = [tool.name if hasattr(tool, "name") else tool.get("name", "") for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        if isinstance(messages[-1], ToolMessage):
            weather = json.loads(messages[-1].content)
            answer = (f"北京{weather['current']['condition']}，温度{weather['current']['temperature_c']}℃。"
                      if weather.get("status") == "ok" else "暂时没能取得天气，请稍后重试。")
            reply = AIMessage(content=answer)
        elif not self.has_city:
            reply = AIMessage(content="你想查询哪个城市的天气？")
        else:
            reply = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "北京", "country_code": "CN"}, "id": "weather-one"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["standard", "deep"])
@pytest.mark.parametrize("available", [True, False])
async def test_real_agent_weather_tool_loop_keeps_nonmedical_routing_and_real_error_status(monkeypatch, mode, available):
    import httpx
    from backend.app.mcp.tools.weather import WeatherTool
    from backend.tests.test_weather_tool import FORECAST, PLACE

    def handler(request):
        if not available:
            return httpx.Response(401, json={})
        return httpx.Response(200, json={"results": [PLACE]} if "geocoding" in request.url.host else FORECAST)

    class WeatherRegistry(Registry):
        async def invoke(self, name, arguments, context):
            self.calls.append((name, arguments))
            assert name == "get_weather"
            return SimpleNamespace(result=await WeatherTool(transport=httpx.MockTransport(handler)).invoke(arguments, context))

    registry = WeatherRegistry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    model = WeatherModel()
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    result = await runtime.run(message="北京现在什么天气？", model_id="auto", conversation_id="weather-regression",
                               user_id=1, case_id=None, history=[{"role": "assistant", "message": "登录过期，无法查天气。"}],
                               on_event=lambda *args: None, reasoning_mode=mode)
    assert [name for name, args in registry.calls] == ["get_weather"]
    assert "get_weather" in model.available_tools
    assert result["agent_trace"][0]["evidence_sufficient"] is None
    assert "登录" not in result["response"]
    assert result["tool_calls"][0]["status"] == ("success" if available else "error")
    if available:
        assert "10.5" in result["response"]
        assert "api.open-meteo.com" in result["response"]


@pytest.mark.anyio
async def test_real_agent_can_request_city_without_searching_medical_sources(monkeypatch):
    registry = Registry()
    runtime = OpenSourceAgentRuntime(registry, EgressGuard())
    model = WeatherModel(has_city=False)
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    result = await ask(runtime, "现在什么天气？")
    assert registry.calls == []
    assert "城市" in result["response"]
    assert "登录" not in result["response"]
    assert "get_weather" in model.available_tools


@pytest.mark.anyio
async def test_stale_session_errors_are_not_replayed_from_client_history(monkeypatch):
    runtime = OpenSourceAgentRuntime(Registry(), EgressGuard())
    model = WeatherModel(has_city=False)
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    await runtime.run(message="现在什么天气？", model_id="auto", conversation_id="legacy-client-history", user_id=1,
                      case_id=None, on_event=lambda *args: None, history=[
                          {"role": "assistant", "message": "操作未完成：登录已过期，请重新登录。"},
                          {"role": "assistant", "message": "我这边确实拿不到天气：登录状态过期导致工具失败。"},
                          {"role": "assistant", "message": "Python 可以用 sorted 排序。"},
                      ])
    contents = [str(item.content) for item in model.seen_messages[0] if isinstance(item, AIMessage)]
    assert contents == ["Python 可以用 sorted 排序。"]


@pytest.mark.anyio
async def test_stale_session_errors_are_removed_from_existing_checkpoints(monkeypatch):
    class StaleModel(RecordingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="我仍然无法查天气：登录状态已过期。"))])

    runtime = OpenSourceAgentRuntime(Registry(), EgressGuard())
    monkeypatch.setattr(runtime, "_model", lambda model_id: StaleModel())
    await ask(runtime, "现在什么天气？")
    model = WeatherModel(has_city=False)
    monkeypatch.setattr(runtime, "_model", lambda model_id: model)
    result = await ask(runtime, "现在什么天气？")
    assert "城市" in result["response"]
    assert not any("登录状态已过期" in str(item.content) for item in model.seen_messages[0] if isinstance(item, AIMessage))


def test_session_cleanup_keeps_user_questions_and_normal_auth_explanations():
    assert not module._stale_session_failure({"role": "user", "message": "我无法操作，登录状态已过期怎么办？"})
    assert not module._stale_session_failure(AIMessage(content="登录过期一般需要重新验证，Python 可以捕获 HTTP 401。"))
