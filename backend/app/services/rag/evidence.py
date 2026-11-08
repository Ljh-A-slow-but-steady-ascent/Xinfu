"""Shared, request-scoped evidence tools and citation checks. No model calls."""
from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .retriever import LocalRagRetriever


class SearchEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(min_length=1, max_length=3)
    top_k: int = Field(default=8, ge=1, le=10)


class ReadEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    include_neighbors: bool = True


def tool_specs() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": "search_evidence",
            "description": "检索本地医学证据摘要。自主改写或拆分查询；可以多查询并行召回。分数只表示相关性。",
            "parameters": SearchEvidenceInput.model_json_schema(),
        }},
        {"type": "function", "function": {
            "name": "read_evidence",
            "description": "阅读已搜索证据的完整段落及相邻上下文，以核对适用条件、冲突和引用原文。",
            "parameters": ReadEvidenceInput.model_json_schema(),
        }},
    ]


class EvidenceSession:
    """Each run owns its observations/cache; clinical queries are not cached globally."""

    def __init__(self, retriever: LocalRagRetriever) -> None:
        self.retriever = retriever
        self.observed: dict[str, dict[str, Any]] = {}
        self.cache: dict[str, dict[str, Any]] = {}
        self.trace: list[dict[str, Any]] = []
        self.backends: list[str] = []

    def invoke(self, name: str, arguments: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        if name == "search_evidence":
            parsed = SearchEvidenceInput.model_validate(arguments)
            queries = list(dict.fromkeys(item.strip() for item in parsed.queries if item.strip()))
            if not queries or any(len(item) > 1500 for item in queries):
                raise ValueError("检索查询须非空且每条不超过1500字")
            params = {"queries": queries, "top_k": parsed.top_k}
        elif name == "read_evidence":
            parsed = ReadEvidenceInput.model_validate(arguments)
            params = parsed.model_dump()
            if any(item not in self.observed for item in parsed.evidence_ids):
                raise ValueError("只能阅读本次搜索已返回的证据ID")
        else:
            raise ValueError("不支持的证据工具")
        key = name + json.dumps(params, ensure_ascii=False, sort_keys=True)
        cached = key in self.cache
        if cached:
            result = copy.deepcopy(self.cache[key])
        else:
            if name == "search_evidence":
                chunks = self.retriever.search_many(**params, timeout=timeout)
                self.backends.extend(str(item["retrieval_backend"]) for item in chunks
                                     if str(item["retrieval_backend"]) not in self.backends)
                # Model sees only a short preview until it chooses to read more.
                chunks = [dict(item, content=str(item["content"])[:600],
                               content_truncated=len(str(item["content"])) > 600) for item in chunks]
            else:
                chunks = self.retriever.read(**params)
                # Bound context even if a future corpus contains very large paragraphs.
                chunks = [dict(item, content=str(item["content"])[:2400],
                               content_truncated=len(str(item["content"])) > 2400) for item in chunks][:15]
            for item in chunks:
                previous = self.observed.get(str(item["id"]))
                if previous is None or len(str(item["content"])) >= len(str(previous["content"])):
                    self.observed[str(item["id"])] = item
            result = {"chunks": chunks, "assessment": "候选证据，须结合本次问题判断适用性；相关性分数不代表证据充分"}
            self.cache[key] = copy.deepcopy(result)
        self.trace.append({"step": len(self.trace) + 1, "tool": name, "arguments": params,
                           "evidence_ids": [item["id"] for item in result["chunks"]], "cached": cached})
        return result


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(min_length=1, max_length=200)
    quote: str = Field(min_length=2, max_length=1000)


class SupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1500)
    citations: list[Citation] = Field(min_length=1, max_length=5)


class EvidenceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[SupportedClaim] = Field(default_factory=list, max_length=8)
    sufficient: bool = False
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    clarification_question: str = Field(default="", max_length=500)


def validate_answer(answer: EvidenceAnswer, observed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify attribution and exact quoted spans; entailment remains a model assessment."""
    claims: list[dict[str, Any]] = []
    errors: list[str] = []
    used: list[str] = []
    for claim in answer.claims:
        valid = True
        for citation in claim.citations:
            chunk = observed.get(citation.evidence_id)
            if chunk is None or citation.quote not in str(chunk.get("content", "")):
                errors.append(f"引用原文无法核验：{citation.evidence_id}")
                valid = False
            elif chunk.get("authoritative") is not True:
                errors.append(f"本地规则材料不能支持医学结论：{citation.evidence_id}")
                valid = False
        if valid:
            claims.append(claim.model_dump())
            used.extend(item.evidence_id for item in claim.citations)
    sufficient = bool(answer.sufficient and claims and not errors and not answer.missing_information
                      and not answer.clarification_question)
    text = "\n\n".join(
        item["text"] + " " + " ".join(f"[证据:{citation['evidence_id']}]" for citation in item["citations"])
        for item in claims
    )
    gaps = list(answer.missing_information)
    if errors:
        gaps.append("部分结论的引用未通过校验，已移除这些结论")
    if not sufficient:
        text += ("\n\n" if text else "") + "当前知识库证据不足以完整回答本次问题。"
    if gaps:
        text += "\n待补充：" + "；".join(gaps)
    if answer.clarification_question:
        text += "\n需要澄清：" + answer.clarification_question
    return {"answer": text, "claims": claims, "evidence_sufficient": sufficient,
            "information_gaps": gaps, "clarification_question": answer.clarification_question,
            "used_evidence_ids": list(dict.fromkeys(used)),
            "citation_validation": {"valid": not errors, "errors": errors,
                "quote_spans_verified": bool(claims),
                "semantic_support": "model_assessed" if claims else "not_assessed"}}
