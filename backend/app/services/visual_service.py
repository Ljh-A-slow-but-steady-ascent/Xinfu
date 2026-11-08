from __future__ import annotations

from pathlib import Path
from typing import Protocol

from backend.app.care.schemas import CandidatePrediction, ClinicalMetadata, VisualEvidence
from backend.app.database import CaseRepository
from backend.app.services.case_service import CaseService
from backend.app.services.vision.quality import assess_image_quality
from backend.app.services.vision_worker import worker


class VisualRunner(Protocol):
    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict: ...


class FrozenDermFMSubprocessRunner:
    """Runs the trained adapter through the shared, isolated model worker."""

    def predict(self, image_path: Path, clinical_metadata: ClinicalMetadata) -> dict:
        return worker.predict(image_path, clinical_metadata.model_dump(mode="json"))


class VisualService:
    def __init__(
        self,
        repository: CaseRepository | None = None,
        case_service: CaseService | None = None,
        runner: VisualRunner | None = None,
    ) -> None:
        self.repository = repository or CaseRepository()
        self.case_service = case_service or CaseService(self.repository)
        self.runner = runner or FrozenDermFMSubprocessRunner()

    def analyze(self, case_id: str, clinical_metadata: ClinicalMetadata) -> VisualEvidence:
        image_path = self.case_service.get_local_image_path(case_id)
        quality, _ = assess_image_quality(image_path)
        raw = self.runner.predict(image_path, clinical_metadata)
        warnings = [
            "候选分数来自图像与结构化病例的联合分析，用于当前候选集合内的相对排序。",
            "候选分数不是经临床校准的患病概率，不能替代医生检查或病理诊断。",
        ]
        result = VisualEvidence(
            case_id=case_id,
            model=str(raw.get("model_name", "local_multimodal_classifier")),
            candidate_set=str(raw.get("candidate_set", "pad_clinical_multimodal_v1")),
            top_predictions=[CandidatePrediction.model_validate(item) for item in raw["top_predictions"]],
            image_quality=quality,
            supports_triage=True,
            latency_ms=float(raw.get("latency_ms", 0.0)),
            local_only=True,
            warnings=warnings,
            prediction_mode=str(raw.get("prediction_mode", "supervised_multimodal_pad")),
            input_modality=str(raw.get("input_modality", "clinical_image+structured_text")),
            clinical_context=clinical_metadata,
        )
        self.repository.save_visual(case_id, result)
        return result

    def get(self, case_id: str) -> VisualEvidence:
        raw = self.repository.get_visual(case_id)
        if raw is None:
            raise KeyError(f"visual result not found: {case_id}")
        return VisualEvidence.model_validate(raw)
