from __future__ import annotations

import pytest

from backend.app.services import knowledge_catalog, rare_disease_service
from backend.app.services.rag.retriever import LocalRagRetriever


class EvidenceService:
    def __init__(self):
        self.calls = []

    def retrieve_patient_context(self, **kwargs):
        self.calls.append(kwargs)
        return {"chunks": [{"id": "A", "title": "证据", "content": "病例需要医生进一步复核。"}],
                "answer": "已核对的解释 [证据:A]", "evidence_sufficient": False,
                "information_gaps": ["尚缺检查资料"], "agent_trace": [{"tool": "search_evidence"}],
                "retrieval": {"backend": "test_hybrid", "mode": "agentic", "model_calls": 2}}


@pytest.mark.parametrize("mode", ["image_text", "zero_shot", "text_only"])
def test_rare_infer_uses_agent_result_and_preserves_vision_modes(monkeypatch, mode):
    knowledge = EvidenceService()
    monkeypatch.setattr(rare_disease_service, "_knowledge", knowledge)
    monkeypatch.setattr(rare_disease_service, "_run_worker", lambda *args: {
        "top_predictions": [{"label": "melanoma", "score": 0.8}], "input_modality": mode})
    result = rare_disease_service.infer(image=None if mode == "text_only" else b"fake-image",
        filename="test.png", text="皮损近期变化", mode=mode)
    assert knowledge.calls[0]["visual_candidates"] == ["melanoma"]
    assert result["rag"]["answer"] == "已核对的解释 [证据:A]"
    assert result["rag"]["agent_trace"]
    assert result["rag"]["backend"] == "test_hybrid"
    assert result["rag"]["items"] == result["rag"]["chunks"]
    assert result["rag"]["evidence_sufficient"] is False
    assert result["reasoning_mode"] == ("native_zero_shot_clip" if mode == "zero_shot" else "supervised_task_head")




def test_knowledge_snapshot_ids_match_runtime_corpus(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_ADVANCED", "0")
    runtime = LocalRagRetriever()
    items = knowledge_catalog._all_items()
    ids = [chunk["id"] for item in items for chunk in item["chunks"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(runtime.by_id)
    assert "evidence_threshold" not in knowledge_catalog.overview()["retrieval"]


def test_domain_catalog_totals_and_overview_use_the_same_distribution(monkeypatch):
    monkeypatch.setenv("DERMAGUARD_RAG_ADVANCED", "0")
    result = knowledge_catalog.overview()
    expected = [("皮肤肿瘤", 188000), ("炎症与过敏", 165000), ("感染性皮肤病", 142000),
                ("诊断与病理", 120000), ("药物与安全", 110000), ("色素性疾病", 92000),
                ("毛发与甲病", 78000), ("免疫与罕见病", 70000), ("儿童皮肤病", 62000), ("照护与转诊", 55000)]
    assert [(item["label"], item["chunks"]) for item in result["domain_distribution"]] == expected
    assert result["domain_count"] == 10
    assert result["domain_total_chunks"] == 1_082_000
    assert result["domain_total_chunks"] == sum(item["chunks"] for item in result["domain_distribution"])
    assert result["chunk_count"] == len(LocalRagRetriever().by_id)
