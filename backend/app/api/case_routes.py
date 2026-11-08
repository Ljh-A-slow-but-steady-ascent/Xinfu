from __future__ import annotations

from functools import partial
import json
from typing import Any, Literal

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from backend.app.api.dependencies import Runtime
from backend.app.api.responses import http_error, lesion_image_response
from backend.app.api.schemas import (
    AnalysisResult,
    CaseCreated,
    CaseDeleteRequest,
    CaseUpdateRequest,
    PatientCreate,
)
from backend.app.auth import require_user
from backend.app.care.schemas import CarePathway, ClinicalMetadata, TriageDecision, VisualEvidence
from backend.app.services.dermfm_candidate_sets import DISPLAY_LABELS_ZH
from backend.app.services.privacy_service import PrivacyReport


router = APIRouter(tags=["cases"])


@router.post("/cases", response_model=CaseCreated)
async def create_case(
    app: Runtime,
    image: UploadFile = File(...),
    mode: Literal["patient", "clinician"] = Form("clinician"),
    retention_mode: Literal["session", "local"] = Form("local"),
    patient_id: str | None = Form(default=None),
    lesion_site: str = Form(default="未填写"),
    chief_complaint: str = Form(default="未填写"),
    clinical_metadata_json: str = Form(default=""),
) -> CaseCreated:
    try:
        if not patient_id and os.getenv("DERMAGUARD_AUTH_DISABLED", "0") != "1":
            raise ValueError("图像分析前必须先关联患者信息")
        profile: ClinicalMetadata | None = None
        if patient_id:
            patient = app.services.repository.get_patient(patient_id)
            if patient is None:
                raise KeyError(f"patient not found: {patient_id}")
            if not clinical_metadata_json.strip():
                raise ValueError("请在图像分析前完整填写结构化病例信息")
            profile = ClinicalMetadata.model_validate(json.loads(clinical_metadata_json))
            expected_gender = {"男": "MALE", "女": "FEMALE"}.get(patient.sex)
            if expected_gender is None:
                raise ValueError("当前多模态模型的官方病例格式仅接受男性或女性")
            if profile.age != patient.age or profile.gender != expected_gender:
                raise ValueError("结构化病例的年龄或性别与患者档案不一致")
        content = await image.read(10 * 1024 * 1024 + 1)
        case_id = app.services.cases.create(
            content,
            image.content_type or "application/octet-stream",
            mode=mode,
            retention_mode=retention_mode,
        )
        if patient_id:
            assert profile is not None
            app.services.repository.link_case_patient(
                case_id,
                patient_id,
                lesion_site.strip() or "未填写",
                chief_complaint.strip() or "未填写",
                profile.model_dump(mode="json"),
            )
        return CaseCreated(case_id=case_id, mode=mode, retention_mode=retention_mode)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/patients")
async def create_patient(payload: PatientCreate, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        patient = app.services.repository.create_patient(
            name=payload.name.strip(), sex=payload.sex, age=payload.age,
            outpatient_no=payload.outpatient_no.strip(), phone=payload.phone.strip() or None,
            medical_history=payload.medical_history.strip(), allergy_history=payload.allergy_history.strip(),
            created_by=int(user["id"]),
        )
        return app.services.repository.patient_public(patient)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases")
async def list_cases(
    app: Runtime, user: dict = Depends(require_user),
    query: str = Query(default="", max_length=120),
    sex: str = Query(default="", max_length=20),
    lesion_site: str = Query(default="", max_length=120),
    min_age: int | None = Query(default=None, ge=0, le=130),
    max_age: int | None = Query(default=None, ge=0, le=130),
    diagnosis: str = Query(default="", max_length=120),
    triage_level: str = Query(default="", max_length=40),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    normalized_diagnosis = next(
        (key for key, label in DISPLAY_LABELS_ZH.items() if diagnosis.strip() and diagnosis.strip() in label),
        diagnosis,
    )
    items = app.services.repository.list_cases(
        query=query, sex=sex, lesion_site=lesion_site, min_age=min_age,
        max_age=max_age, diagnosis=normalized_diagnosis, triage_level=triage_level,
        folder_id=folder_id, user_id=int(user["id"]),
    )
    locations = app.services.repository.workspace_locations(int(user["id"]), "case")
    return [{**item, "folder_id": locations.get(item["case_id"])} for item in items]


@router.get("/cases/{case_id}/patient")
async def get_case_patient(case_id: str, app: Runtime) -> dict[str, Any]:
    patient = app.services.repository.get_case_patient(case_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="该病例尚未关联患者")
    return patient


@router.put("/cases/{case_id}")
async def update_case(case_id: str, payload: CaseUpdateRequest, app: Runtime) -> dict[str, Any]:
    try:
        return app.services.repository.update_case_record(
            case_id, payload.model_dump(exclude_unset=True, mode="json")
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str, payload: CaseDeleteRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    if payload.confirm_case_id != case_id:
        raise HTTPException(status_code=409, detail="病例编号确认不匹配")
    try:
        app.services.cases.delete(case_id, int(user["id"]))
        return {"status": "deleted", "case_id": case_id, "reports_and_image_deleted": True}
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases/{case_id}/image")
async def get_case_image(
    case_id: str, app: Runtime, user: dict = Depends(require_user),
) -> Response:
    try:
        if not app.services.repository.case_belongs_to_user(case_id, int(user["id"])):
            raise KeyError("病例不存在或当前账号无权访问")
        path = app.services.cases.get_local_image_path(case_id)
        return lesion_image_response(path)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/cases/{case_id}/analyze", response_model=AnalysisResult)
async def analyze_case(case_id: str, app: Runtime) -> AnalysisResult:
    try:
        result = await to_thread.run_sync(partial(app.services.analysis.run, case_id))
        return AnalysisResult.model_validate(result)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases/{case_id}/visual", response_model=VisualEvidence)
async def get_visual(case_id: str, app: Runtime) -> VisualEvidence:
    try:
        return app.services.visual.get(case_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases/{case_id}/triage", response_model=TriageDecision)
async def get_triage(case_id: str, app: Runtime) -> TriageDecision:
    try:
        return app.services.triage.get(case_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases/{case_id}/care-pathway", response_model=CarePathway)
async def get_care_pathway(case_id: str, app: Runtime) -> CarePathway:
    try:
        return app.services.care.get(case_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/cases/{case_id}/privacy", response_model=PrivacyReport)
async def get_privacy(case_id: str, app: Runtime) -> PrivacyReport:
    try:
        return app.services.privacy.report(case_id)
    except Exception as exc:
        raise http_error(exc) from exc
