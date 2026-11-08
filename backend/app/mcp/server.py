from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.app.mcp.factory import create_tool_registry
from backend.app.mcp.schemas import ToolContext
from backend.app.services.container import ServiceContainer, create_service_container


def create_mcp_server(services: ServiceContainer | None = None) -> FastMCP:
    services = services or create_service_container()
    registry = create_tool_registry(services)
    server = FastMCP(
        name="xinfu-local-tools",
        instructions=(
            "Xinfu public-web search and reading, evidence, care-pathway, guideline, and privacy tools. "
            "Never request or return raw medical image bytes or patient identity."
        ),
    )

    @server.tool(structured_output=True)
    async def get_visual_evidence(case_id: str) -> dict[str, Any]:
        """Read persisted local visual evidence; raw images are never returned."""
        call = await registry.invoke(
            "get_visual_evidence",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def get_triage_result(case_id: str) -> dict[str, Any]:
        """Read the immutable deterministic TriagePolicy result."""
        call = await registry.invoke(
            "get_triage_result",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def search_guideline(
        query: str, case_id: str | None = None, top_k: int = 3
    ) -> dict[str, Any]:
        """Search only the local triage and referral guideline collection."""
        call = await registry.invoke(
            "search_guideline",
            {"query": query, "case_id": case_id, "top_k": top_k},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def search_evidence(queries: list[str], top_k: int = 8) -> dict[str, Any]:
        """Search local evidence snippets; the calling agent chooses rewritten queries."""
        call = await registry.invoke("search_evidence", {"queries": queries, "top_k": top_k})
        return call.result

    @server.tool(structured_output=True)
    async def read_evidence(evidence_ids: list[str], include_neighbors: bool = True) -> dict[str, Any]:
        """Read complete evidence paragraphs and their neighboring context."""
        call = await registry.invoke("read_evidence", {
            "evidence_ids": evidence_ids, "include_neighbors": include_neighbors})
        return call.result

    @server.tool(structured_output=True)
    async def get_care_pathway(case_id: str) -> dict[str, Any]:
        """Read the next-care pathway bound to the policy result."""
        call = await registry.invoke(
            "get_care_pathway",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def get_privacy_report(case_id: str) -> dict[str, Any]:
        """Read audited facts about outbound case data and image locality."""
        call = await registry.invoke(
            "get_privacy_report",
            {"case_id": case_id},
            ToolContext(case_id=case_id, channel="mcp", destination="mcp_client"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def web_search(query: str, top_k: int = 5, time_range: str | None = None) -> dict[str, Any]:
        """Search public webpages for information across topics."""
        call = await registry.invoke(
            "web_search", {"query": query, "top_k": top_k, "time_range": time_range},
            ToolContext(channel="http", destination="public_search_engines"),
        )
        return call.result

    @server.tool(structured_output=True)
    async def read_webpage(url: str, max_chars: int = 14000, focus: str | None = None) -> dict[str, Any]:
        """Read public webpage content to verify search results or a supplied URL."""
        return (await registry.invoke("read_webpage", {"url": url, "max_chars": max_chars, "focus": focus})).result

    @server.tool(structured_output=True)
    async def get_weather(city: str, country_code: str | None = None, days: int = 1) -> dict[str, Any]:
        """Read current weather and forecast for a user-specified city."""
        return (await registry.invoke("get_weather", {"city": city, "country_code": country_code, "days": days},
                                     ToolContext(channel="http", destination="api.open-meteo.com"))).result

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
