from __future__ import annotations

from functools import partial
from typing import Annotated
import asyncio
from collections import OrderedDict
import time
from uuid import uuid4

from anyio import to_thread
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.app.auth import require_user
from backend.app.services import rare_disease_service as rare

router = APIRouter(prefix="/v1/rare-disease", tags=["analysis"])
User = Annotated[dict, Depends(require_user)]
_jobs: OrderedDict[str, dict] = OrderedDict()
_tasks: set[asyncio.Task] = set()


async def _image_bytes(image: UploadFile | None) -> bytes | None:
    raw = await image.read(10 * 1024 * 1024 + 1) if image else None
    if raw is not None and len(raw) > 10 * 1024 * 1024:
        raise HTTPException(422, "图片不能超过 10 MB")
    return raw


def _validate_input(mode: str, text: str, raw: bytes | None) -> None:
    if mode not in {"image_only", "text_only", "image_text", "zero_shot"}:
        raise HTTPException(422, "推理模式不受支持")
    if mode != "text_only" and not raw:
        raise HTTPException(422, "请上传病灶图片")
    if mode == "text_only" and not text.strip():
        raise HTTPException(422, "请输入病例描述")


def _prune_jobs() -> None:
    for key, job in list(_jobs.items()):
        if job["status"] != "RUNNING" and time.monotonic() - job["created"] > 3600:
            del _jobs[key]
    while len(_jobs) >= 64:
        finished = next((key for key, job in _jobs.items() if job["status"] != "RUNNING"), None)
        if finished is None:
            break
        del _jobs[finished]


async def _analyze_job(job: dict, **arguments) -> None:
    try:
        job["result"] = await to_thread.run_sync(partial(rare.infer, **arguments))
        job["status"] = "COMPLETED"
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = "FAILED"


@router.post("/jobs", status_code=202)
async def create_rare_job(
    user: User, mode: str = Form("image_text"), text: str = Form(""),
    top_k: int = Form(5), image: UploadFile | None = File(None),
):
    raw = await _image_bytes(image)
    _validate_input(mode, text, raw)
    _prune_jobs()
    if sum(job["status"] == "RUNNING" for job in _jobs.values()) >= 4:
        raise HTTPException(429, "图片分析队列已满，请稍后重试")
    job_id = uuid4().hex
    job = {"job_id": job_id, "owner": int(user["id"]), "status": "RUNNING", "created": time.monotonic()}
    _jobs[job_id] = job
    task = asyncio.create_task(_analyze_job(
        job, image=raw, filename=image.filename if image else "", text=text,
        mode=mode, top_k=max(1, min(top_k, 10)),
    ))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job_id": job_id, "status": job["status"]}


@router.get("/jobs/{job_id}")
async def get_rare_job(job_id: str, user: User):
    job = _jobs.get(job_id)
    if job is None or job["owner"] != int(user["id"]):
        raise HTTPException(404, "图片分析任务不存在或已过期")
    return {key: job[key] for key in ("job_id", "status", "result", "error") if key in job}


@router.post("/infer")
async def rare_disease_infer(
    _: User, mode: str = Form("image_text"), text: str = Form(""), top_k: int = Form(5), image: UploadFile | None = File(None)
):
    raw = await _image_bytes(image)
    _validate_input(mode, text, raw)
    try:
        return await to_thread.run_sync(partial(
            rare.infer, image=raw, filename=image.filename if image else "",
            text=text, mode=mode, top_k=max(1, min(top_k, 10)),
        ))
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
