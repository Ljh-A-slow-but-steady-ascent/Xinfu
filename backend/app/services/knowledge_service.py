from __future__ import annotations

from typing import Any

from backend.app.care.schemas import GuidelineReference
from backend.app.services.privacy import EgressGuard
from backend.app.services.privacy.egress import sanitize_text
from backend.app.services.rag import LocalRagRetriever
from backend.app.services.rag.agent import AgenticRag
from backend.app.services.rag.evidence import EvidenceSession
from backend.app.services.rag.llm import RagLLM
from backend.app.services.rag.retriever import build_dynamic_query


class KnowledgeService:
    def __init__(self, retriever: LocalRagRetriever | None = None, *,
                 guard: EgressGuard | None = None, llm: RagLLM | None = None) -> None:
        self.retriever = retriever or LocalRagRetriever()
        self.guard = guard or EgressGuard()
        self.llm = llm

    def _safe_query(self, query: str, case_id: str | None = None) -> str:
        repository = self.guard.repository
        if repository is not None and case_id:
            record = repository.get_case_patient(case_id) or {}
            for key in ("name", "patient_name", "phone", "outpatient_no"):
                value = str(record.get(key) or "").strip()
                if value:
                    query = query.replace(value, "[身份已隐藏]")
        return sanitize_text(query)[0]

    def retrieve(self, query: str, top_k: int = 5, *, case_id: str | None = None) -> dict[str, Any]:
        result = AgenticRag(self.retriever, self.llm or RagLLM(self.guard)).run(
            self._safe_query(query, case_id), top_k=top_k)
        result["evidence_assembly"] = self.assemble_evidence(result["chunks"], assessment=result)
        return result

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Compatibility entry: business queries use the same autonomous agent."""
        return self.retrieve(query, top_k=top_k)["chunks"]

    def retrieve_patient_context(
        self, *, user_query: str = "", visual_candidates: list[str] | None = None,
        positive_symptoms: list[str] | None = None, risk_reasons: list[str] | None = None,
        duration: str | None = None, topic: str = "鉴别依据 检查依据 风险分级 转诊指南",
        top_k: int = 5, case_id: str | None = None,
    ) -> dict[str, Any]:
        query = build_dynamic_query(user_query=user_query, visual_candidates=visual_candidates,
            positive_symptoms=positive_symptoms, risk_reasons=risk_reasons, duration=duration, topic=topic)
        return self.retrieve(query, top_k=top_k, case_id=case_id)


    def search_evidence(self, queries: list[str], top_k: int = 8) -> dict[str, Any]:
        """Raw tool for an existing agent; never nests another model/agent."""
        result = EvidenceSession(self.retriever).invoke("search_evidence", {
            "queries": [sanitize_text(item)[0] for item in queries], "top_k": top_k})
        return {**result, "evidence_assembly": self._pending_assessment(),
                "retrieval": {"backend": "；".join(dict.fromkeys(str(item["retrieval_backend"])
                              for item in result["chunks"])) or "未返回候选证据", "mode": "agent_tool"}}

    def read_evidence(self, evidence_ids: list[str], include_neighbors: bool = True) -> dict[str, Any]:
        # MCP callers carry their own observation state. Only public corpus IDs
        # are accepted; this API never resolves arbitrary filesystem paths.
        chunks = self.retriever.read(evidence_ids, include_neighbors=include_neighbors)
        chunks = [dict(item, content=str(item["content"])[:2400],
                       content_truncated=len(str(item["content"])) > 2400) for item in chunks][:15]
        return {"chunks": chunks, "evidence_assembly": self._pending_assessment()}

    @staticmethod
    def _pending_assessment() -> dict[str, Any]:
        # Do not duplicate each snippet under several categories in model context.
        return {"evidence_sufficient": False, "assessment": "待Agent评估",
                "answer_policy": "仅依据实际看到的证据作有限解释；分数不表示充分性"}

    def references(self, chunks: list[dict[str, Any]]) -> list[GuidelineReference]:
        return [GuidelineReference(id=str(item["id"]), title=str(item["title"]), source=str(item["source"]))
                for item in chunks if item.get("verified_citation")]

    @staticmethod
    def assemble_evidence(chunks: list[dict[str, Any]], *, assessment: dict[str, Any] | None = None) -> dict[str, Any]:
        categories: dict[str, list[dict[str, Any]]] = {name: [] for name in (
            "风险依据", "鉴别依据", "检查依据", "转诊依据", "冲突证据", "信息缺口")}
        for item in chunks:
            for category in item.get("evidence_types", [item.get("evidence_type", "信息缺口")]):
                categories.setdefault(str(category), []).append({
                    "evidence_id": item["id"], "statement": item["content"],
                    "citation": item.get("citation", {}), "source": item["source"], "score": item.get("score", 0),
                })
        sufficient = bool(assessment and assessment.get("evidence_sufficient"))
        return {"categories": categories, "evidence_sufficient": sufficient,
                "information_gaps": assessment.get("information_gaps", []) if assessment else [],
                "assessment": ("model_assessed_with_quote_checks" if assessment.get("retrieval", {}).get("stop_reason") == "completed"
                               else "未完成Agent评估") if assessment else "待Agent评估",
                "answer_policy": "仅引用已核验片段作有限解释" if sufficient else "证据尚未充分核验，不得补充未提供的医学事实"}
