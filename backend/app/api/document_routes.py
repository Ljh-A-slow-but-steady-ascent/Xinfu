from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from backend.app.api.dependencies import Runtime
from backend.app.api.responses import http_error
from backend.app.api.schemas import AgentDocumentUpdateRequest
from backend.app.auth import require_user


router = APIRouter(tags=["documents"])


@router.get("/agent/documents")
async def list_agent_documents(
    app: Runtime, user: dict = Depends(require_user),
    conversation_id: str = Query(default="", max_length=100),
    folder_id: str = Query(default="", max_length=64),
) -> list[dict[str, Any]]:
    items = app.services.repository.list_agent_documents(
        user_id=int(user["id"]), conversation_id=conversation_id or None
    )
    if folder_id:
        ids = set(app.services.repository.folder_resource_ids(int(user["id"]), folder_id, "document"))
        items = [item for item in items if item["document_id"] in ids]
    locations = app.services.repository.workspace_locations(int(user["id"]), "document")
    return [{**item, "folder_id": locations.get(item["document_id"])} for item in items]


@router.post("/agent/documents/upload")
async def upload_agent_document(
    app: Runtime, user: dict = Depends(require_user),
    file: UploadFile = File(...), conversation_id: str = Form(default="workspace"),
    case_id: str = Form(default=""), folder_id: str = Form(default=""),
) -> dict[str, Any]:
    allowed = {".md", ".txt", ".json", ".csv"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail="仅支持 Markdown、TXT、JSON 和 CSV 文本文档")
    raw = await file.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="文档不能超过 2 MB")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="文档必须使用 UTF-8 编码") from exc
    item = app.services.repository.create_agent_document(
        user_id=int(user["id"]), conversation_id=conversation_id or "workspace",
        case_id=case_id or None, title=os.path.splitext(file.filename or "未命名文档")[0],
        filename=file.filename or "document.md", content=content,
    )
    if folder_id:
        app.services.repository.move_workspace_item(int(user["id"]), "document", item["document_id"], folder_id)
    return item


@router.put("/agent/documents/{document_id}")
async def update_agent_document(
    document_id: str, request: AgentDocumentUpdateRequest,
    app: Runtime, user: dict = Depends(require_user),
) -> dict[str, Any]:
    try:
        return app.services.repository.update_agent_document(
            document_id, int(user["id"]), content=request.content,
            title=request.title.strip() or None,
        )
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/agent/documents/{document_id}")
async def delete_agent_document(
    document_id: str, app: Runtime, user: dict = Depends(require_user),
) -> dict[str, Any]:
    if not app.services.repository.delete_agent_document(document_id, int(user["id"])):
        raise HTTPException(status_code=404, detail="Agent 文档不存在")
    return {"status": "deleted", "document_id": document_id, "case_and_report_retained": True}


@router.get("/agent/documents/{document_id}")
async def get_agent_document_detail(
    document_id: str, app: Runtime, user: dict = Depends(require_user)
) -> dict[str, Any]:
    item = app.services.repository.get_agent_document(document_id, int(user["id"]))
    if item is None:
        raise HTTPException(status_code=404, detail="Agent 文档不存在")
    return item


@router.get("/agent/documents/{document_id}/download")
async def download_agent_document(
    document_id: str, app: Runtime, user: dict = Depends(require_user)
) -> Response:
    item = app.services.repository.get_agent_document(document_id, int(user["id"]))
    if item is None:
        raise HTTPException(status_code=404, detail="Agent 文档不存在")
    encoded = quote(item["filename"])
    return Response(
        item["content"].encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )
