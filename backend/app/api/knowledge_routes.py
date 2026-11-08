from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.app.auth import require_user
from backend.app.services import knowledge_catalog as kb

router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])
User = Annotated[dict, Depends(require_user)]


@router.get("/overview")
async def knowledge_overview(_: User) -> dict:
    return kb.overview()


@router.get("/items")
async def knowledge_items(
    _: User, source: str = Query(""), risk: str = Query(""), query: str = Query("")
) -> dict:
    return {"data_source": "filesystem_snapshot", "items": kb.items(source, risk, query)}


@router.get("/items/{document_id}")
async def knowledge_item(document_id: str, _: User) -> dict:
    value = kb.item(document_id)
    if value is None:
        raise HTTPException(404, "资料不存在")
    return {"data_source": "filesystem_snapshot", **value}


@router.post("/uploads")
async def knowledge_upload(_: User, file: UploadFile = File(...)) -> dict:
    allowed = {".md", ".txt", ".json", ".csv", ".jpg", ".jpeg", ".png", ".webp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(422, "资料上传仅支持 Markdown、TXT、JSON、CSV、JPG、PNG 和 WebP")
    raw = await file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(422, "资料上传文件不能超过 5 MB")
    return kb.save_upload(file.filename or "upload.bin", raw)
