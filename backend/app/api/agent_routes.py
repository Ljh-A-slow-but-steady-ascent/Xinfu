from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.app.api.dependencies import Runtime
from backend.app.api.responses import http_error
from backend.app.api.schemas import (
    AgentActionRequest,
    AgentIntakeExtractionRequest,
    AgentIntakeExtractionResponse,
    AgentResumeRequest,
)
from backend.app.auth import require_user
from backend.app.chat import chat_model_catalog
from backend.app.harness import AgentIntake, HarnessResponse
from backend.app.services.clinical_extraction_service import field_labels


router = APIRouter(tags=["agent"])


def _conversation_history(raw: str) -> list[dict[str, str]]:
    if not raw.strip():
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("对话上下文必须是列表")
    history: list[dict[str, str]] = []
    for item in value[-12:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        message = str(item.get("message", "")).strip()
        if message:
            history.append({"role": str(item["role"]), "message": message[:4000]})
    return history


@router.post("/agent/intake/extract", response_model=AgentIntakeExtractionResponse)
async def extract_agent_intake(
    payload: AgentIntakeExtractionRequest, app: Runtime
) -> AgentIntakeExtractionResponse:
    try:
        result = app.services.clinical_extraction.extract_agent_intake(
            narrative=payload.narrative, model_id=payload.model
        )
        return AgentIntakeExtractionResponse(
            intake=result.intake,
            complete=not result.missing_fields,
            missing_fields=result.missing_fields,
            missing_field_labels=field_labels(result.missing_fields),
            uncertain_fields=result.uncertain_fields,
            provider=result.provider,
            model=result.model,
            sanitized_fields=result.sanitized_fields,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs", response_model=HarnessResponse)
async def create_agent_run(
    app: Runtime,
    user: dict = Depends(require_user),
    message: str = Form(default=""),
    case_id: str = Form(default=""),
    model: str = Form(default="auto"),
    detail_level: Literal["concise", "detailed"] = Form(default="concise"),
    reasoning_mode: Literal["standard", "deep"] = Form(default="standard"),
    intake_json: str = Form(default=""),
    conversation_id: str = Form(default=""),
    history_json: str = Form(default=""),
    image: UploadFile | None = File(default=None),
) -> HarnessResponse:
    try:
        intake = AgentIntake.model_validate_json(intake_json) if intake_json.strip() else None
        image_bytes = await image.read(10 * 1024 * 1024 + 1) if image else None
        return await app.harness.start(
            user_id=int(user["id"]), message=message,
            case_id=case_id.strip() or None, model=model,
            detail_level=detail_level, image_bytes=image_bytes,
            reasoning_mode=reasoning_mode,
            image_content_type=image.content_type if image else None, intake=intake,
            conversation_id=conversation_id.strip() or None,
            conversation_history=_conversation_history(history_json),
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs/dispatch", response_model=HarnessResponse)
async def dispatch_agent_run(
    app: Runtime,
    user: dict = Depends(require_user),
    message: str = Form(default=""),
    case_id: str = Form(default=""),
    model: str = Form(default="auto"),
    detail_level: Literal["concise", "detailed"] = Form(default="concise"),
    reasoning_mode: Literal["standard", "deep"] = Form(default="standard"),
    intake_json: str = Form(default=""),
    conversation_id: str = Form(default=""),
    history_json: str = Form(default=""),
    image: UploadFile | None = File(default=None),
) -> HarnessResponse:
    """Return a run id immediately; the UI polls persisted graph events."""
    try:
        intake = AgentIntake.model_validate_json(intake_json) if intake_json.strip() else None
        image_bytes = await image.read(10 * 1024 * 1024 + 1) if image else None
        normalized_case_id = case_id.strip() or None
        initial = app.harness.create_run(
            user_id=int(user["id"]), case_id=normalized_case_id, message=message,
            reasoning_mode=reasoning_mode,
        )
        task = asyncio.create_task(app.harness.execute(
            run_id=initial.run_id, user_id=int(user["id"]), message=message,
            case_id=normalized_case_id, model=model, detail_level=detail_level,
            reasoning_mode=reasoning_mode,
            image_bytes=image_bytes,
            image_content_type=image.content_type if image else None, intake=intake,
            conversation_id=conversation_id.strip() or initial.run_id,
            conversation_history=_conversation_history(history_json),
        ))
        app.tasks.add(task)
        task.add_done_callback(app.tasks.discard)
        task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)
        return initial
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/agent/runs/{run_id}", response_model=HarnessResponse)
async def get_agent_run(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        record = app.services.repository.get_agent_run_record(run_id)
        if record is None:
            raise KeyError("Agent任务不存在")
        if record.user_id != int(user["id"]):
            raise PermissionError("无权读取其他医生的Agent任务")
        node_labels = {
            "intake": "正在读取请求…", "route_request": "正在选择执行路径…",
            "general_knowledge_retrieval": "正在选择知识源并检索…",
            "medical_knowledge_retrieval": "正在检索本地医学证据…",
            "public_web_search": "正在检索公开网络资料…",
            "clinical_chat": "正在读取病例并规划检索…",
            "multimodal_analysis": "正在执行本地联合分析…",
            "confirm_case_creation": "病例资料已准备，等待医生确认。",
            "confirm_report_generation": "报告生成任务已准备，等待医生确认。",
            "confirm_agent_action": "Agent 已生成操作预览，等待医生确认。",
            "completed": "任务已完成。", "failed": "任务执行失败。",
        }
        current = app.services.repository.get_agent_run(run_id) or {}
        message = node_labels.get(current.get("current_node"), "正在处理任务…")
        last_result = current.get("state", {}).get("last_result", {})
        if current.get("status") == "COMPLETED" and last_result.get("kind") == "chat":
            message = last_result.get("chat", {}).get("response") or message
        elif (
            current.get("status") == "WAITING_CONFIRM"
            and last_result.get("kind") == "chat"
        ):
            message = last_result.get("chat", {}).get("response") or message
        elif current.get("status") == "WAITING_CONFIRM":
            message = "资料已准备完成，请核对后确认执行。"
        return app.harness.response(run_id, message)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs/{run_id}/confirm", response_model=HarnessResponse)
async def confirm_agent_action(
    run_id: str, payload: AgentActionRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> HarnessResponse:
    try:
        return await app.harness.confirm(run_id, int(user["id"]), payload.action_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs/{run_id}/reject", response_model=HarnessResponse)
async def reject_agent_action(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        return app.harness.reject(run_id, int(user["id"]))
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs/{run_id}/pause", response_model=HarnessResponse)
async def pause_agent_run(
    run_id: str, app: Runtime, user: dict = Depends(require_user)
) -> HarnessResponse:
    try:
        return app.harness.pause(run_id, int(user["id"]))
    except Exception as exc:
        raise http_error(exc) from exc


@router.post("/agent/runs/{run_id}/resume", response_model=HarnessResponse)
async def resume_agent_run(
    run_id: str, payload: AgentResumeRequest, app: Runtime,
    user: dict = Depends(require_user),
) -> HarnessResponse:
    try:
        return await app.harness.resume(
            run_id, int(user["id"]), payload.message, payload.model, payload.detail_level,
            reasoning_mode=payload.reasoning_mode,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.get("/chat/models")
async def chat_models() -> list[dict[str, Any]]:
    return chat_model_catalog()
