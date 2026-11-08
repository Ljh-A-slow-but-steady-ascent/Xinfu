from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response

from backend.app.api.dependencies import Runtime
from backend.app.api.responses import http_error, lesion_image_response
from backend.app.api.schemas import ReportGenerateRequest, ReportUpdateRequest
from backend.app.auth import require_user
from backend.app.services.report_service import ReportService


router = APIRouter(tags=["reports"])


@router.post("/cases/{case_id}/reports")
async def generate_report(
    case_id: str,
    request: ReportGenerateRequest,
    app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        patient = app.services.repository.get_case_patient(case_id)
        if patient is None:
            raise KeyError("该病例尚未关联患者")
        clinical_payload = {
            "visual": app.services.repository.get_visual(case_id) or {},
            "triage": app.services.repository.get_triage(case_id) or {},
            "care_pathway": app.services.repository.get_care_pathway(case_id) or {},
            "clinical_profile": patient.get("clinical_profile", {}),
        }
        generated = ReportService(app.guard).generate(
            case_id=case_id,
            patient=patient,
            clinical_payload=clinical_payload,
            model_id=request.model,
        )
        record = app.services.repository.create_report(
            case_id=case_id,
            patient_id=patient["patient_id"],
            content=generated.content,
            structured=generated.structured,
            provider=generated.provider,
            model=generated.model,
            created_by=int(user["id"]),
        )
        report = app.services.repository.get_report(record.report_id)
        if report is None:
            raise RuntimeError("报告保存失败")
        return report
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/reports")
async def list_reports(
    app: Runtime, user: dict = Depends(require_user),
    query: str = Query(default="", max_length=120),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    items = app.services.repository.list_reports(query=query, user_id=int(user["id"]))
    if folder_id:
        ids = set(app.services.repository.folder_resource_ids(int(user["id"]), folder_id, "report"))
        items = [item for item in items if item["report_id"] in ids]
    locations = app.services.repository.workspace_locations(int(user["id"]), "report")
    return [{**item, "folder_id": locations.get(item["report_id"])} for item in items]


@router.get("/reports/{report_id}")
async def get_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


@router.put("/reports/{report_id}")
async def update_report(
    report_id: str,
    request: ReportUpdateRequest,
    app: Runtime,
    user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return app.services.repository.update_report(
            report_id, request.content, int(user["id"]), request.status
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    if not app.services.repository.delete_report(report_id, int(user["id"])):
        raise HTTPException(status_code=404, detail="报告不存在")
    return {"status": "deleted", "report_id": report_id, "case_and_image_retained": True}


@router.get("/reports/{report_id}/image")
async def get_report_image(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> Response:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    path = app.services.cases.get_local_image_path(report["case_id"])
    return lesion_image_response(path)


@router.get("/reports/{report_id}/download")
async def download_report(report_id: str, app: Runtime, user: dict = Depends(require_user)) -> PlainTextResponse:
    report = app.services.repository.get_report_for_user(report_id, int(user["id"]))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return PlainTextResponse(
        report["content"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={report['report_id']}.md"},
    )
