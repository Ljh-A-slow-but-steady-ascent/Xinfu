"""Compose the authenticated application API without duplicating business logic."""
from fastapi import APIRouter, Depends

from . import agent_routes, case_routes, document_routes, report_routes, workspace_routes
from backend.app.auth import require_user

router = APIRouter(prefix="/v1", dependencies=[Depends(require_user)])
for module in (agent_routes, case_routes, document_routes, report_routes, workspace_routes):
    router.include_router(module.router)
