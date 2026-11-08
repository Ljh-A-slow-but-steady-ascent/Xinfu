from __future__ import annotations

import ipaddress
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
import safehttpx
from lxml import html
from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy import EgressGuard, EgressRequest

logger = logging.getLogger(__name__)
MAX_PAGE_BYTES = 1_500_000


class PublicTransport(safehttpx.AsyncSecureTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host
        host_header = request.headers.get("Host", hostname)
        request.url = request.url.copy_with(host=self.verified_ip)
        request.headers["Host"] = host_header
        request.extensions["sni_hostname"] = hostname
        return await httpx.AsyncHTTPTransport.handle_async_request(self, request)


def public_url(value: str) -> str:
    url = urlsplit(value.strip())
    host = (url.hostname or "").rstrip(".").lower()
    if url.scheme not in {"http", "https"} or not host or url.username is not None or url.password is not None:
        raise ValueError("public HTTP URL required")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or url.port not in {None, 80, 443}:
        raise ValueError("public website required")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("public website required")
    return urlunsplit((url.scheme, url.netloc, url.path or "/", url.query, url.fragment))


def extract_text(body: bytes, content_type: str, anchor: str = "") -> tuple[str, str]:
    if "html" not in content_type:
        return "网页内容", body.decode("utf-8", errors="replace").strip()
    root = html.fromstring(body)
    title = " ".join(root.xpath("//title/text()")) or "网页内容"
    sections = root.xpath("//*[@id=$anchor]", anchor=unquote(anchor)) if anchor else []
    if sections:
        root = sections[0].getparent() if sections[0].tag == "dt" else sections[0]
    for element in root.xpath("//script|//style|//noscript|//svg|//nav|//footer|//header|//form"):
        element.drop_tree()
    main = root.xpath(".//main|.//article|.//*[@role='main']")
    text = (main[0] if main else root).text_content()
    text = "\n".join(re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines() if line.strip())
    return title.strip()[:240], text.strip()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=8, max_length=2048)
    max_chars: int = Field(default=14000, ge=1000, le=20000)
    focus: str | None = Field(default=None, max_length=200, description="需要查阅长网页的特定部分时，填写要定位的关键词或方法名。")


class ReadWebpageTool:
    name = "read_webpage"
    description = "读取公开网页正文，用于核对搜索结果、官方文档和用户提供的网页；返回正文及来源链接。"

    def __init__(self, guard: EgressGuard | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.guard = guard or EgressGuard()
        self.transport = transport

    def input_schema(self) -> dict[str, Any]:
        return Input.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = Input.model_validate(arguments)
        try:
            url = public_url(request.url)
            for _ in range(4):
                checked = self.guard.enforce(EgressRequest(channel="http", destination=urlsplit(url).hostname or "",
                    payload={"url": url}, case_id=context.case_id, tool_name=self.name))
                url = public_url(str(checked.sanitized_payload["url"]))
                verified_ip = await safehttpx.async_validate_url(urlsplit(url).hostname)
                async with httpx.AsyncClient(transport=self.transport or PublicTransport(verified_ip), timeout=10,
                    headers={"User-Agent": "Xinfu/1.0 (public webpage reader)"}) as client:
                    async with client.stream("GET", url, follow_redirects=False) as response:
                        if response.is_redirect:
                            url = public_url(urljoin(url, response.headers["location"]))
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "text/html").lower()
                        if not any(value in content_type for value in ("text/", "application/json", "application/xhtml+xml")):
                            return {"status": "unsupported_format", "url": url, "message": "这个链接没有可读取的网页正文，可查找其他网页来源。"}
                        body = bytearray()
                        truncated = False
                        async for chunk in response.aiter_bytes(chunk_size=16384):
                            body.extend(chunk[:MAX_PAGE_BYTES - len(body)])
                            if len(body) >= MAX_PAGE_BYTES:
                                truncated = True
                                break
                title, text = extract_text(bytes(body), content_type, urlsplit(url).fragment)
                match = text.casefold().find(request.focus.casefold()) if request.focus else -1
                start = max(0, match - 300)
                return {"status": "ok" if text else "empty", "title": title, "url": url,
                        "content": text[start:start + request.max_chars], "truncated": truncated or len(text) > request.max_chars,
                        "focus_found": match >= 0 if request.focus else None,
                        "source_scope": "外部公开网页", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "results": [{"title": title, "url": url}]}
            raise ValueError("redirect limit exceeded")
        except Exception as error:
            logger.warning("Public webpage unavailable: %s", type(error).__name__)
            return {"status": "unavailable", "message": "暂时没有读取到网页正文，可以换一个来源继续查询。"}
