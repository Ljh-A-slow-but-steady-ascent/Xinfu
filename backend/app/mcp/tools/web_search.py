from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from ddgs import DDGS
from lxml import etree
from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy import EgressGuard, EgressRequest
from .webpage import public_url

logger = logging.getLogger(__name__)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, description="自主改写的网页搜索词，可包含地点、日期、官方站点或任务关键词。")
    top_k: int = Field(default=5, ge=1, le=5)
    time_range: Literal["d", "w", "m", "y"] | None = Field(default=None, description="需要时限制最近一天、一周、一月或一年。")


class WebSearchTool:
    name = "web_search"
    description = "搜索公开互联网网页，返回标题、链接和摘要；适用于实时信息、官方文档及各种普通问题，按需继续读取网页原文。"

    def __init__(self, guard: EgressGuard | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.guard = guard or EgressGuard()
        self.transport = transport

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    @staticmethod
    def _search(query: str, top_k: int, time_range: str | None) -> list[dict]:
        proxy = (os.getenv("DDGS_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
                 or os.getenv("ALL_PROXY") or os.getenv("all_proxy"))
        with DDGS(proxy=proxy, timeout=8) as search:
            return search.text(query, max_results=top_k, region="us-en", timelimit=time_range,
                               backend=os.getenv("XINFU_WEB_SEARCH_BACKENDS", "bing,yahoo"))

    async def _feed_search(self, query: str, top_k: int, time_range: str | None) -> list[dict]:
        params = {"q": query, "format": "rss"}
        if time_range:
            params["filters"] = f'ex1:"ez{ {"d": 1, "w": 2, "m": 3, "y": 5}[time_range]}"'
        async with httpx.AsyncClient(transport=self.transport, timeout=8, headers={"User-Agent": "Mozilla/5.0 Xinfu/1.0"}) as client:
            response = await client.get("https://www.bing.com/search", params=params)
            response.raise_for_status()
        root = etree.fromstring(response.content, parser=etree.XMLParser(resolve_entities=False, no_network=True))
        return [{"title": item.findtext("title"), "href": item.findtext("link"), "body": item.findtext("description")}
                for item in root.findall(".//item")[:top_k]]

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        checked = self.guard.enforce(EgressRequest(
            channel="http", destination="public_search_engines", payload={"query": request.query},
            case_id=context.case_id, tool_name=self.name,
        ))
        query = str(checked.sanitized_payload["query"])
        try:
            try:
                rows = await asyncio.to_thread(self._search, query, request.top_k, request.time_range)
            except Exception:
                rows = []
            if not rows:
                rows = await self._feed_search(query, request.top_k, request.time_range)
            results = []
            seen = set()
            for row in rows:
                try:
                    url = public_url(str(row.get("href") or row.get("url") or ""))
                except ValueError:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                results.append({"title": str(row.get("title") or url)[:240], "url": url,
                                "snippet": str(row.get("body") or row.get("snippet") or "")[:1600],
                                "source": urlsplit(url).hostname})
            return {"status": "ok" if results else "no_results", "query": query, "results": results[:request.top_k],
                    "source_scope": "外部公开网页", "retrieved_at": datetime.now(timezone.utc).isoformat()}
        except Exception as error:
            logger.warning("Public search unavailable: %s", type(error).__name__)
            return {"status": "unavailable", "query": query, "results": [], "message": "暂时没有取得搜索结果，可以调整关键词后再查。"}
