"""Subprocess bridge for the trained PAD image-text classifier."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# All model/tokenizer artifacts are already permanent local files. Prevent
# availability checks from introducing proxy/network delays during inference.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PAD_MULTIMODAL_HEAD = Path(
    os.getenv(
        "DERMAGUARD_PAD_MULTIMODAL_CHECKPOINT",
        str(ROOT / "artifacts" / "training" / "pad_multimodal" / "image_text" / "best.pt"),
    )
)
PAD_HEAD_ROOT = PAD_MULTIMODAL_HEAD.parent.parent
PAD_LABELS = {
    "NEV": "nevus",
    "BCC": "basal cell carcinoma",
    "ACK": "actinic keratosis",
    "SEK": "seborrheic keratosis",
    "SCC": "squamous cell carcinoma",
    "MEL": "melanoma",
}
RARE_ZERO_SHOT_LABELS = {
    "cutaneous mastocytosis": "皮肤肥大细胞增生症",
    "dermatomyositis": "皮肌炎",
    "porphyria cutanea tarda": "迟发性皮肤卟啉症",
    "pityriasis rubra pilaris": "毛发红糠疹",
    "epidermolysis bullosa": "大疱性表皮松解症",
    "pemphigus vulgaris": "寻常型天疱疮",
    "bullous pemphigoid": "大疱性类天疱疮",
    "calciphylaxis": "钙化防御",
    "cutaneous sarcoidosis": "皮肤结节病",
    "lichen sclerosus": "硬化性苔藓",
}

_MODEL_CACHE: dict = {}


def _prepare_mode(mode: str) -> None:
    if _MODEL_CACHE.get("mode") != mode:
        _MODEL_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_pad_model(mode: str):
    from pad_multimodal.core import DermFMPADClassifier, load_dermfm_components

    _prepare_mode(mode)
    if not _MODEL_CACHE:
        checkpoint = PAD_HEAD_ROOT / mode / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PAD checkpoint not found: {checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision, text_encoder, tokenizer = load_dermfm_components()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("mode") != mode:
            raise RuntimeError(f"Configured PAD checkpoint mode mismatch: {saved.get('mode')} != {mode}")
        model = DermFMPADClassifier(vision, text_encoder, mode=mode, num_classes=int(saved["num_classes"]))
        model.load_trainable_checkpoint(checkpoint)
        model.to(device).eval()
        _MODEL_CACHE.update(mode=mode, model=model, tokenizer=tokenizer, device=device,
                            label_names={int(key): value for key, value in saved["label_names"].items()})
    return _MODEL_CACHE


def predict_zero_shot(image_path: Path, text: str, top_k: int) -> dict:
    """Native DermFM CLIP-style image/text cosine matching without a task head."""
    from pad_multimodal.inference import preprocess_image
    from pad_multimodal.core import load_dermfm_model

    if image_path is None:
        raise ValueError("zero-shot 模式需要上传图片")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _prepare_mode("zero_shot")
    if not _MODEL_CACHE:
        model, tokenizer = load_dermfm_model()
        model.to(device).eval()
        _MODEL_CACHE.update(mode="zero_shot", model=model, tokenizer=tokenizer)
    model, tokenizer = _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]
    image = preprocess_image(image_path).to(device)
    labels = list(RARE_ZERO_SHOT_LABELS)
    prompts = tokenizer([f"This is a clinical skin image of {label}." for label in labels]).to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(prompts).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    values, indices = probabilities.topk(min(top_k, len(labels)))
    return {"model_name": "DermFM-Zero-Open", "candidate_set": "rare_dermatology_zero_shot_v1", "top_predictions": [{"label": RARE_ZERO_SHOT_LABELS[labels[int(index)]], "label_en": labels[int(index)], "score": float(value)} for value, index in zip(values, indices)], "prediction_mode": "zero_shot_clip_cosine", "input_modality": "clinical_image+text_prompts", "prompt_context": text, "checkpoint": str(ROOT / "weights" / "DermFM-Zero-Open" / "open_clip_pytorch_model.bin"), "latency_ms": (time.perf_counter() - started) * 1000.0}


def predict_pad_multimodal(image_path: Path | None, metadata: dict, top_k: int, mode: str = "image_text") -> dict:
    """Run the trained official image-text fusion adapter on one clinical case."""
    from pad_multimodal.core import build_pad_prompt
    from pad_multimodal.inference import preprocess_image

    if mode == "zero_shot":
        return predict_zero_shot(image_path, str(metadata.get("chief_complaint", "")), top_k)
    if mode not in {"image_only", "text_only", "image_text"}:
        raise ValueError("mode must be image_only, text_only, image_text or zero_shot")
    checkpoint = PAD_HEAD_ROOT / mode / "best.pt"
    started = time.perf_counter()
    prompt = build_pad_prompt(metadata)
    cached = load_pad_model(mode)
    model, tokenizer, device = cached["model"], cached["tokenizer"], cached["device"]
    image = preprocess_image(image_path).to(device) if image_path is not None else None
    tokens = tokenizer([prompt]).to(device)
    with torch.inference_mode():
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model(image, tokens).float().softmax(dim=1)[0].cpu()
        else:
            probabilities = model(image, tokens).float().softmax(dim=1)[0].cpu()
    label_names = cached["label_names"]
    values, indices = probabilities.topk(min(top_k, len(label_names)))
    predictions = [
        {
            "label": PAD_LABELS.get(str(label_names[int(index)]), str(label_names[int(index)])),
            "score": float(value),
        }
        for value, index in zip(values, indices)
    ]
    if not all(math.isfinite(item["score"]) for item in predictions):
        raise RuntimeError("PAD multimodal head returned a non-finite score")
    return {
        "model_name": "local_multimodal_classifier",
        "candidate_set": "pad_clinical_multimodal_v1",
        "top_predictions": predictions,
        "prediction_mode": "supervised_multimodal_pad",
        "input_modality": {"image_only": "clinical_image", "text_only": "structured_text", "image_text": "clinical_image+structured_text"}[mode],
        "checkpoint": str(checkpoint),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["image_only", "text_only", "image_text", "zero_shot"], default="image_text")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("action") == "warmup":
                    load_pad_model("image_text")
                    result = {"status": "ready"}
                else:
                    result = predict_pad_multimodal(
                        Path(request["image"]) if request.get("image") else None,
                        request.get("metadata", {}), int(request.get("top_k", 3)),
                        request.get("mode", "image_text"),
                    )
            except Exception as exc:
                result = {"error": str(exc)}
            print("DERMAGUARD_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
        return
    metadata = json.loads(sys.stdin.read())
    result = predict_pad_multimodal(Path(args.image) if args.image else None, metadata, args.top_k, args.mode)
    print("DERMAGUARD_RESULT=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
