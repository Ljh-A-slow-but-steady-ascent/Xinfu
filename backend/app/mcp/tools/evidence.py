from __future__ import annotations

from functools import partial
from anyio import to_thread
from typing import Any

from backend.app.mcp.schemas import ToolContext
from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.rag.evidence import ReadEvidenceInput, SearchEvidenceInput


class SearchEvidenceTool:
    name = "search_evidence"
    description = "Search local medical evidence snippets with agent-written queries; no nested model calls."

    def __init__(self, service: KnowledgeService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return SearchEvidenceInput.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = SearchEvidenceInput.model_validate(arguments)
        return await to_thread.run_sync(partial(self.service.search_evidence, **request.model_dump()))


class ReadEvidenceTool:
    name = "read_evidence"
    description = "Read full local evidence paragraphs and neighboring context by known evidence ID."

    def __init__(self, service: KnowledgeService) -> None:
        self.service = service

    def input_schema(self) -> dict[str, Any]:
        return ReadEvidenceInput.model_json_schema()

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        request = ReadEvidenceInput.model_validate(arguments)
        return await to_thread.run_sync(partial(self.service.read_evidence, **request.model_dump()))
