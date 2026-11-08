from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.care.schemas import CarePathway, ClinicalMetadata, TriageDecision, VisualEvidence
from backend.app.services.privacy_service import PrivacyReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseCreated(StrictModel):
    case_id: str
    mode: Literal["patient", "clinician"]
    retention_mode: Literal["session", "local"]
    raw_image_local_only: bool = True


class PatientCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    sex: Literal["男", "女", "其他"]
    age: int = Field(ge=0, le=130)
    outpatient_no: str = Field(min_length=1, max_length=80)
    phone: str = Field(default="", max_length=40)
    medical_history: str = Field(default="", max_length=3000)
    allergy_history: str = Field(default="", max_length=2000)


class AgentIntakeExtractionRequest(StrictModel):
    narrative: str = Field(min_length=10, max_length=5000)
    model: str = Field(default="auto", max_length=120)


class AgentIntakeExtractionResponse(StrictModel):
    intake: dict[str, Any]
    complete: bool
    missing_fields: list[str]
    missing_field_labels: list[str]
    uncertain_fields: list[str]
    provider: str
    model: str
    privacy_status: Literal["SAFE"] = "SAFE"
    sanitized_fields: list[str] = Field(default_factory=list)


class AgentDocumentUpdateRequest(StrictModel):
    content: str = Field(min_length=1, max_length=60000)
    title: str = Field(default="", max_length=160)


class CaseUpdateRequest(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    sex: Literal["男", "女"] | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    outpatient_no: str | None = Field(default=None, min_length=1, max_length=80)
    phone: str | None = Field(default=None, max_length=40)
    medical_history: str | None = Field(default=None, max_length=3000)
    allergy_history: str | None = Field(default=None, max_length=2000)
    lesion_site: str | None = Field(default=None, max_length=120)
    chief_complaint: str | None = Field(default=None, max_length=5000)
    clinical_profile: ClinicalMetadata | None = None


class CaseDeleteRequest(StrictModel):
    confirm_case_id: str


class FolderCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    parent_id: str | None = Field(default=None, max_length=64)


class WorkspaceMoveRequest(StrictModel):
    resource_type: Literal["case", "report", "document"]
    resource_id: str = Field(min_length=1, max_length=100)
    folder_id: str | None = Field(default=None, max_length=64)


class ReportGenerateRequest(StrictModel):
    model: str = Field(default="local-template", max_length=120)


class ReportUpdateRequest(StrictModel):
    content: str = Field(min_length=1, max_length=50_000)
    status: Literal["draft", "final"] = "final"


class AgentActionRequest(StrictModel):
    action_id: str = Field(min_length=1, max_length=80)


class AgentResumeRequest(StrictModel):
    message: str = Field(min_length=1, max_length=5000)
    model: str = Field(default="auto", max_length=120)
    detail_level: Literal["concise", "detailed"] = "concise"
    reasoning_mode: Literal["standard", "deep"] | None = None


class PerformanceMetrics(StrictModel):
    vision_model_latency_ms: float = Field(ge=0)
    visual_stage_wall_ms: float = Field(ge=0)
    triage_latency_ms: float = Field(ge=0)
    rag_latency_ms: float = Field(ge=0)
    care_pathway_latency_ms: float = Field(ge=0)
    total_pretriage_latency_ms: float = Field(ge=0)


class AnalysisResult(StrictModel):
    case_id: str
    visual: VisualEvidence
    triage: TriageDecision
    care_pathway: CarePathway
    guidelines: list[dict[str, Any]]
    rag: dict[str, Any] = Field(default_factory=dict)
    privacy: PrivacyReport
    performance: PerformanceMetrics
