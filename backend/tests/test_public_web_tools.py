from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from backend.app.harness.open_source_agent import OpenSourceAgentRuntime
from backend.app.mcp.schemas import ToolContext
from backend.app.mcp.tools import webpage
from backend.app.mcp.tools.web_search import WebSearchTool
from backend.app.mcp.tools.webpage import ReadWebpageTool, extract_text, public_url
from backend.app.services.privacy import EgressGuard


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_search_cross_topic_results_are_deduplicated_and_private_urls_filtered(monkeypatch):
    captured = []

    def search(query, top_k, time_range):
        captured.append((query, top_k, time_range))
        return [
            {"title": "Python docs", "href": "https://docs.python.org/3/", "body": "Python reference"},
            {"title": "duplicate", "href": "https://docs.python.org/3/"},
            {"href": "http://127.0.0.1/private"},
            {"title": "Museum", "href": "https://www.chnmuseum.cn/", "body": "Opening hours"},
        ]

    monkeypatch.setattr(WebSearchTool, "_search", staticmethod(search))
    result = await WebSearchTool().invoke({"query": "官方资料 13800138000", "top_k": 5, "time_range": "w"}, ToolContext())
    assert result["status"] == "ok"
    assert [item["source"] for item in result["results"]] == ["docs.python.org", "www.chnmuseum.cn"]
    assert captured == [("官方资料 [REDACTED_PHONE]", 5, "w")]


@pytest.mark.anyio
async def test_search_uses_public_feed_when_primary_engine_fails(monkeypatch):
    def fail(*args):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(WebSearchTool, "_search", staticmethod(fail))

    def handler(request):
        assert request.url.host == "www.bing.com"
        assert request.url.params["q"] == "博物馆 官网"
        assert request.url.params["filters"] == 'ex1:"ez2"'
        return httpx.Response(200, text='''<rss><channel><item><title>国家博物馆</title>
            <link>https://www.chnmuseum.cn/cg/</link><description>参观服务</description>
            </item></channel></rss>''')

    result = await WebSearchTool(transport=httpx.MockTransport(handler)).invoke({"query": "博物馆 官网", "time_range": "w"}, ToolContext())
    assert result["status"] == "ok"
    assert result["results"][0]["url"] == "https://www.chnmuseum.cn/cg/"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://localhost/", "http://127.0.0.1/", "http://[::1]/",
    "http://169.254.169.254/latest", "http://192.168.1.2/", "http://service.internal/",
    "https://user:pass@example.org/", "https://example.org:8501/",
])
def test_reader_accepts_only_public_http_urls(url):
    with pytest.raises(ValueError):
        public_url(url)


@pytest.mark.anyio
async def test_reader_checks_dns_and_redirect_before_following(monkeypatch):
    resolved = []
    requested = []

    async def validate(host):
        resolved.append(host)
        return "93.184.216.34"

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secrets"})

    monkeypatch.setattr(webpage.safehttpx, "async_validate_url", validate)
    result = await ReadWebpageTool(transport=httpx.MockTransport(handler)).invoke({"url": "https://example.org/start"}, ToolContext())
    assert result["status"] == "unavailable"
    assert resolved == ["example.org"]
    assert requested == ["https://example.org/start"]
    assert "secrets" not in json.dumps(result)


def test_reader_extracts_anchor_section_without_page_navigation_or_scripts():
    title, text = extract_text(b'''<html><title>Python docs</title><nav>Site menu</nav><main>
        <dl><dt id="Path.read_text">Path.read_text(encoding=None)</dt><dd>Read text.<script>bad()</script></dd></dl>
        <p>Other method</p></main></html>''', "text/html", "Path.read_text")
    assert title == "Python docs"
    assert "Path.read_text(encoding=None)" in text
    assert "Read text." in text
    assert all(value not in text for value in ("Site menu", "bad()", "Other method"))


@pytest.mark.anyio
async def test_reader_focuses_long_body_and_limits_model_context(monkeypatch):
    async def validate(host):
        return "93.184.216.34"

    monkeypatch.setattr(webpage.safehttpx, "async_validate_url", validate)
    body = "Introduction " * 2000 + "Opening hours: 09:00–17:00. " + "Details " * 3000
    result = await ReadWebpageTool(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body, headers={"content-type": "text/plain"})
    )).invoke({"url": "https://example.org/", "focus": "Opening hours", "max_chars": 1000}, ToolContext())
    assert result["status"] == "ok"
    assert result["focus_found"] is True
    assert "09:00–17:00" in result["content"]
    assert len(result["content"]) == 1000
    assert result["truncated"] is True


@pytest.mark.anyio
async def test_public_tool_cache_and_shared_call_budget_are_per_request():
    calls = []

    class Registry:
        async def invoke(self, name, arguments, context):
            calls.append((name, arguments))
            return SimpleNamespace(result={"status": "ok", "results": []})

    runtime = OpenSourceAgentRuntime(Registry(), EgressGuard())
    kwargs = dict(user_id=1, conversation_id="budget", case_id=None, on_event=lambda *args: None, proposals=[])
    tools = {tool.name: tool for tool in runtime._tools(**kwargs)}
    await tools["web_search"].ainvoke({"query": "same query"})
    await tools["web_search"].ainvoke({"query": "same query"})
    for index in range(5):
        await tools["read_webpage"].ainvoke({"url": f"https://example.org/{index}"})
    blocked = json.loads(await tools["web_search"].ainvoke({"query": "extra query"}))
    assert len(calls) == 6
    assert blocked["status"] == "budget_exhausted"
    fresh = {tool.name: tool for tool in runtime._tools(**kwargs)}
    await fresh["web_search"].ainvoke({"query": "same query"})
    assert len(calls) == 7
