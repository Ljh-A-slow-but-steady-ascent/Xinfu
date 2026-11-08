from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.services.knowledge_service import KnowledgeService
from backend.app.services.vision_worker import worker


_knowledge = KnowledgeService()


def _run_worker(image: bytes | None, filename: str, text: str, mode: str, top_k: int) -> dict:
    temp_path: Path | None = None
    try:
        if image is not None:
            suffix = Path(filename or "lesion.png").suffix.lower() or ".png"
            with tempfile.NamedTemporaryFile(prefix="rare-lesion-", suffix=suffix, delete=False) as handle:
                handle.write(image)
                temp_path = Path(handle.name)
        return worker.predict(temp_path, {"chief_complaint": text}, mode=mode, top_k=top_k)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def infer(*, image: bytes | None, filename: str, text: str, mode: str, top_k: int = 5) -> dict:
    if mode in {"image_only", "image_text", "zero_shot"} and image is None:
        raise ValueError("仅图像模式需要上传图片")
    if mode == "text_only" and not text.strip():
        raise ValueError("仅文本模式需要输入病例描述")
    raw = _run_worker(image, filename, text, mode, top_k)
    predictions = raw.get("top_predictions", [])
    retrieval = _knowledge.retrieve_patient_context(
        user_query=text or "罕见皮肤病病灶辅助鉴别",
        visual_candidates=[str(item.get("label", "")) for item in predictions],
        topic="罕见皮肤病 鉴别依据 检查依据 转诊依据 信息缺口",
        top_k=5,
    )
    package_id = f"RARE-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6].upper()}"
    return {
        "package_id": package_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "rare_disease_supervised_head",
        "checkpoint": raw.get("checkpoint"),
        "mode": mode,
        "input_modality": raw.get("input_modality"),
        "text": text,
        "top_predictions": predictions,
        "rag": {**retrieval, "items": retrieval["chunks"], "backend": retrieval["retrieval"]["backend"]},
        "memory_policy": "explicit_package_selection",
        "reasoning_mode": "native_zero_shot_clip" if mode == "zero_shot" else "supervised_task_head",
    }
