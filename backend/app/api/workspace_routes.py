from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import Runtime
from backend.app.api.responses import http_error
from backend.app.api.schemas import FolderCreateRequest, WorkspaceMoveRequest
from backend.app.auth import require_user


router = APIRouter(tags=["workspace"])


@router.get("/workspace/folders")
async def list_workspace_folders(app: Runtime, user: dict = Depends(require_user)) -> list[dict[str, Any]]:
    return app.services.repository.list_folders(int(user["id"]))


@router.post("/workspace/folders")
async def create_workspace_folder(payload: FolderCreateRequest, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        return app.services.repository.create_folder(int(user["id"]), payload.name.strip(), payload.parent_id)
    except Exception as exc:
        raise http_error(exc) from exc


@router.delete("/workspace/folders/{folder_id}")
async def delete_workspace_folder(folder_id: str, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    if not app.services.repository.delete_folder(int(user["id"]), folder_id):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    return {"status": "deleted", "folder_id": folder_id, "resources_retained": True}


@router.put("/workspace/items")
async def move_workspace_item(payload: WorkspaceMoveRequest, app: Runtime, user: dict = Depends(require_user)) -> dict[str, Any]:
    try:
        return app.services.repository.move_workspace_item(
            int(user["id"]), payload.resource_type, payload.resource_id, payload.folder_id
        )
    except Exception as exc:
        raise http_error(exc) from exc
