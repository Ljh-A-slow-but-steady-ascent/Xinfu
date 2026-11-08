"""Shared API error and image responses."""
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response


def http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


LESION_IMAGE_PLACEHOLDER = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='960' height='600'>"
    "<rect width='100%' height='100%' fill='#f4f8fb'/>"
    "<rect x='24' y='24' width='912' height='552' fill='none' stroke='#aac3df' stroke-width='3' stroke-dasharray='14 10' rx='24'/>"
    "<text x='480' y='286' text-anchor='middle' font-family='sans-serif' font-size='30' fill='#6a83a5'>病灶图像待补充</text>"
    "<text x='480' y='330' text-anchor='middle' font-family='sans-serif' font-size='18' fill='#8aa2bd'>此区域为报告预留的病灶图像位置</text>"
    "</svg>"
)


_IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".svg": "image/svg+xml"}


def lesion_image_response(path: Path) -> Response:
    if not path.exists():
        return Response(LESION_IMAGE_PLACEHOLDER, media_type="image/svg+xml")
    return Response(path.read_bytes(), media_type=_IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"))
