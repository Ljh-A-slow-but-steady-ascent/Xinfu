from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class RagConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    top_k: int = Field(3, ge=1, le=10)
    candidate_k: int = Field(8, ge=1, le=16)
    max_tool_calls: int = Field(8, ge=1, le=20)
    max_model_turns: int = Field(10, ge=2, le=24)
    max_seconds: float = Field(45.0, ge=1, le=120)


class VisionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    min_quality: float = Field(0.4, ge=0, le=1)


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    rag: RagConfig = RagConfig()
    vision: VisionConfig = VisionConfig()


@lru_cache(maxsize=1)
def get_config(path: Path | None = None) -> AppConfig:
    config_path = path or BACKEND_ROOT / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
