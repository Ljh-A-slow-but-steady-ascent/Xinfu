"""模型观测：训练指标、样本预测与结构分布资料。

数据来源：
- artifacts/training/pad_multimodal/   PAD-UFES-20 多模态融合（image_only / text_only / image_text）
- artifacts/training/ham7_linear_head/     HAM10000 七分类线性探针
- data/pad_ufes20/official/meta/metadata.csv + data/pad_ufes20/images/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.auth import require_user

router = APIRouter(prefix="/v1/observatory", dependencies=[Depends(require_user)])

ROOT = Path(__file__).resolve().parents[3]
PAD_DIR = ROOT / "artifacts" / "training" / "pad_multimodal"
HAM7_DIR = ROOT / "artifacts" / "training" / "ham7_linear_head"
PAD_IMAGES = ROOT / "data" / "pad_ufes20" / "images"
PAD_META = ROOT / "data" / "pad_ufes20" / "official" / "meta" / "metadata.csv"

PAD_CLASS_ORDER = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
PAD_CLASS_CN = {
    "NEV": "色素痣", "BCC": "基底细胞癌", "ACK": "日光性角化",
    "SEK": "脂溢性角化", "SCC": "鳞状细胞癌", "MEL": "黑色素瘤",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"缺少训练产物：{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"缺少数据文件：{path.name}")
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            rows.append(row)
    return rows


@router.get("/datasets")
def datasets() -> dict[str, Any]:
    """两套真实训练产物概览（多模态融合对比 + 线性探针）。"""
    comparison = _load_json(PAD_DIR / "comparison.json")
    audit = _load_json(PAD_DIR / "dataset_audit.json")
    pad_modes = []
    for mode in ("image_only", "text_only", "image_text"):
        entry = comparison.get(mode) or {}
        test = entry.get("test") or {}
        val = entry.get("validation") or {}
        pad_modes.append({
            "mode": mode,
            "best_epoch": entry.get("best_epoch"),
            "test": {
                "accuracy": test.get("accuracy"), "balanced_accuracy": test.get("balanced_accuracy"),
                "macro_f1": test.get("macro_f1"), "per_class_f1": test.get("per_class_f1"),
                "confusion_matrix": test.get("confusion_matrix"),
            },
            "validation": {
                "accuracy": val.get("accuracy"), "balanced_accuracy": val.get("balanced_accuracy"),
                "macro_f1": val.get("macro_f1"),
            },
        })
    ham_metrics = _load_json(HAM7_DIR / "metrics.json")
    test_metrics = ham_metrics.get("test_metrics") or {}
    ham7 = {
        "name": "HAM10000 七分类线性探针",
        "head_type": ham_metrics.get("head_type"),
        "feature_dim": ham_metrics.get("feature_dim"),
        "temperature": ham_metrics.get("temperature"),
        "abstention_threshold": ham_metrics.get("abstention_threshold"),
        "split_counts": ham_metrics.get("split_counts"),
        "class_names": ham_metrics.get("class_names"),
        "test": {
            "accuracy": test_metrics.get("accuracy"),
            "balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "macro_f1": test_metrics.get("macro_f1"),
            "per_class_f1": test_metrics.get("per_class_f1"),
            "support": test_metrics.get("support"),
            "confusion_matrix": test_metrics.get("confusion_matrix"),
        },
        "validation": ham_metrics.get("validation_selection"),
    }
    return {
        "pad": {
            "name": "PAD-UFES-20 皮肤肿瘤多模态融合",
            "image_count": 2298,
            "class_names": PAD_CLASS_ORDER,
            "class_names_cn": PAD_CLASS_CN,
            "split_sizes": (audit.get("split_sizes") or {}),
            "class_counts": audit.get("class_counts") or {},
            "modes": pad_modes,
        },
        "ham7": ham7,
    }


@router.get("/pad/curves")
def pad_curves(mode: str = Query("image_text")) -> dict[str, Any]:
    """真实训练曲线：来自 training_log.csv，逐 epoch。"""
    if mode not in {"image_only", "text_only", "image_text"}:
        raise HTTPException(status_code=400, detail="mode 取值不合法")
    rows = _read_csv_rows(PAD_DIR / mode / "training_log.csv")
    curves = [
        {
            "epoch": int(float(row.get("epoch") or 0)),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_accuracy": float(row["val_accuracy"]),
            "val_macro_f1": float(row.get("val_macro_f1") or 0.0),
        }
        for row in rows
        if row.get("train_loss") and row.get("val_loss")
    ]
    return {"mode": mode, "points": curves}


@router.get("/pad/cases")
def pad_cases(
    mode: str = Query("image_text"),
    filter: str = Query("all"),
    label: str = Query("all"),
    limit: int = Query(36, le=120),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """真实逐样本预测案例：predictions.csv × metadata.csv。"""
    if mode not in {"image_only", "text_only", "image_text"}:
        raise HTTPException(status_code=400, detail="mode 取值不合法")

    meta_by_image: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(PAD_META):
        img = Path(row.get("img_id") or "").name
        if img:
            meta_by_image[img] = row

    prob_keys = [f"prob_{name}" for name in PAD_CLASS_ORDER]
    cases: list[dict[str, Any]] = []
    for row in _read_csv_rows(PAD_DIR / mode / "predictions.csv"):
        image_path = Path(row.get("image_path") or "")
        img_id = image_path.name
        true_name = row.get("target_name") or ""
        pred_name = row.get("prediction_name") or ""
        correct = true_name == pred_name
        if filter == "correct" and not correct:
            continue
        if filter == "incorrect" and correct:
            continue
        if label != "all" and true_name != label:
            continue
        probs = {}
        for key in prob_keys:
            try:
                probs[key.removeprefix("prob_")] = float(row.get(key) or 0.0)
            except ValueError:
                probs[key.removeprefix("prob_")] = 0.0
        top = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:3]
        meta = meta_by_image.get(img_id, {})
        prompt = (row.get("prompt") or "").strip()
        cases.append({
            "img_id": img_id,
            "true": true_name,
            "true_cn": PAD_CLASS_CN.get(true_name, true_name),
            "pred": pred_name,
            "pred_cn": PAD_CLASS_CN.get(pred_name, pred_name),
            "correct": correct,
            "confidence": top[0][1] if top else 0.0,
            "top3": [{"name": name, "cn": PAD_CLASS_CN.get(name, name), "prob": prob} for name, prob in top],
            "probs": probs,
            "prompt": prompt,
            "meta": {
                "age": meta.get("age") or "",
                "gender": (meta.get("gender") or "").capitalize(),
                "region": (meta.get("region") or "").title(),
                "fitspatrick": meta.get("fitspatrick") or "",
                "diameter": meta.get("diameter_1") or "",
                "biopsed": meta.get("biopsed") or "",
            },
        })
    total = len(cases)
    page = cases[offset : offset + limit]
    return {"mode": mode, "filter": filter, "label": label, "total": total, "cases": page}


@router.get("/pad/image/{img_id}")
def pad_image(img_id: str) -> FileResponse:
    """提供 PAD-UFES-20 原始病灶图像（仅限文件名，防目录穿越）。"""
    safe = Path(img_id).name
    if safe != img_id or not safe.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="非法的图片标识")
    target = (PAD_IMAGES / safe).resolve()
    if target.parent != PAD_IMAGES.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(target)


@router.get("/model-snapshot")
def model_snapshot() -> dict[str, Any]:
    return _load_json(ROOT / "backend" / "app" / "resources" / "model_snapshot.json")
