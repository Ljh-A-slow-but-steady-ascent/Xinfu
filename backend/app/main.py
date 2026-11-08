import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip().strip("'\""))

from backend.app.api.auth_routes import router as auth_router
from backend.app.api.router import router
from backend.app.api.knowledge_routes import router as knowledge_router
from backend.app.api.rare_disease_routes import router as rare_disease_router
from backend.app.api.observatory_routes import router as observatory_router


@asynccontextmanager
async def lifespan(application):
    from backend.app.services.vision_worker import worker
    if os.getenv("XINFU_VISION_PRELOAD", "0") == "1":
        await asyncio.to_thread(worker.warmup)
    yield
    worker.close()


app = FastAPI(
    title="信肤 · Xinfu Clinical Agent",
    version="2.0.0",
    description="面向医生的多模态皮肤病辅助决策、医学证据检索与病例报告服务",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|\[::1\]|(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d+){2,3})(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(router)
app.include_router(knowledge_router)
app.include_router(rare_disease_router)
app.include_router(observatory_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


from backend.app.web import mount_workspace

mount_workspace(app)
