"""Serve the built workspace through the same origin as the API."""
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class WorkspaceFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["X-Xinfu-Workspace"] = "static"
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable"
            if path.startswith("assets/") else "no-cache"
        )
        if Path(path).suffix.lower() == ".zip":
            filename = quote(Path(path).name)
            response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{filename}"
        return response


def mount_workspace(app: FastAPI, directory: Path = FRONTEND_DIST) -> None:
    if (directory / "index.html").is_file():
        files = WorkspaceFiles(directory=directory, html=True)
        app.mount("/", GZipMiddleware(files, minimum_size=1024, compresslevel=5), name="workspace")
