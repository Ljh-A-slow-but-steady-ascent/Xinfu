from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import sys
import time

from fastapi import FastAPI
import httpx
import pytest

from backend.app.services.vision_worker import VisionWorker
from backend.app.web import mount_workspace


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def image_worker(tmp_path):
    program = tmp_path / "worker.py"
    program.write_text('''import json, os, sys, time
counter = 0
for line in sys.stdin:
    payload = json.loads(line)
    time.sleep(payload.get("metadata", {}).get("delay", 0))
    counter += 1
    print("worker log", flush=True)
    print("DERMAGUARD_RESULT=" + json.dumps({"pid": os.getpid(), "counter": counter}), flush=True)
''')
    value = VisionWorker([sys.executable, "-u", str(program)], capacity=3)
    yield value
    value.close()


def test_resident_worker_reuses_process_and_serializes_requests(image_worker):
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: image_worker.predict(None, {"delay": 0.02}), range(3)))
    assert len({item["pid"] for item in results}) == 1
    assert sorted(item["counter"] for item in results) == [1, 2, 3]


def test_worker_timeout_releases_queue_and_next_call_recovers(image_worker):
    with pytest.raises(TimeoutError):
        image_worker.predict(None, {"delay": 2}, timeout=0.1)
    assert image_worker.process is None
    assert image_worker.predict(None, {}, timeout=3)["counter"] == 1


@pytest.mark.anyio
async def test_static_workspace_preserves_api_and_hides_parent_files(tmp_path):
    directory = tmp_path / "dist"; directory.mkdir()
    (directory / "index.html").write_text("<title>信肤</title>")
    (directory / "assets").mkdir()
    source = "export const ready = true;\n" + "// workspace\n" * 200
    (directory / "assets/app.123.js").write_text(source)
    review = directory / "review"; review.mkdir()
    (review / "index.html").write_text("<title>评审病例素材</title>")
    archive = b"PK\x03\x04" + bytes(range(256))
    (review / "xinfu-review-cases.zip").write_bytes(archive)
    (tmp_path / "secret.txt").write_text("private")
    app = FastAPI()
    @app.get("/v1/ping")
    def ping():
        return {"ok": True}
    mount_workspace(app, directory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/")
        assert "信肤" in page.text and page.headers["cache-control"] == "no-cache"
        asset = await client.get("/assets/app.123.js")
        assert "immutable" in asset.headers["cache-control"]
        assert asset.headers["content-encoding"] == "gzip"
        assert asset.text == source
        assert "评审病例素材" in (await client.get("/review/")).text
        package = await client.get("/review/xinfu-review-cases.zip")
        assert package.status_code == 200 and package.content == archive
        assert package.headers["content-type"] == "application/zip"
        assert package.headers["content-disposition"] == "attachment; filename*=UTF-8''xinfu-review-cases.zip"
        partial = await client.get("/review/xinfu-review-cases.zip", headers={"Range": "bytes=0-3"})
        assert partial.status_code == 206 and partial.content == b"PK\x03\x04"
        assert (await client.get("/v1/ping")).json() == {"ok": True}
        assert (await client.get("/%2e%2e/secret.txt")).status_code == 404


@pytest.mark.anyio
async def test_image_jobs_return_immediately_and_enforce_ownership(monkeypatch):
    from backend.app.api import rare_disease_routes as routes
    from backend.app.auth import require_user
    from backend.app.main import app
    def infer(**arguments):
        time.sleep(0.1)
        return {"top_predictions": [{"label": "melanoma", "score": 0.9}]}
    monkeypatch.setattr(routes.rare, "infer", infer)
    app.dependency_overrides[require_user] = lambda: {"id": 71}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post("/v1/rare-disease/jobs", data={"mode": "text_only", "text": "病程"})
            assert created.status_code == 202 and created.json()["status"] == "RUNNING"
            job_id = created.json()["job_id"]
            app.dependency_overrides[require_user] = lambda: {"id": 72}
            assert (await client.get(f"/v1/rare-disease/jobs/{job_id}")).status_code == 404
            app.dependency_overrides[require_user] = lambda: {"id": 71}
            for _ in range(30):
                result = (await client.get(f"/v1/rare-disease/jobs/{job_id}")).json()
                if result["status"] != "RUNNING":
                    break
                await asyncio.sleep(0.02)
            assert result["status"] == "COMPLETED"
            assert result["result"]["top_predictions"][0]["label"] == "melanoma"
    finally:
        app.dependency_overrides.pop(require_user, None)
        routes._jobs.clear()
