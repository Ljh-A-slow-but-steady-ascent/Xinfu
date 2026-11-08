from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends

from backend.app.harness import ClinicalAgentHarness
from backend.app.mcp.factory import create_tool_registry
from backend.app.services.container import ServiceContainer, create_service_container
from backend.app.services.privacy import EgressGuard


@dataclass
class ApiRuntime:
    services: ServiceContainer
    guard: EgressGuard
    harness: ClinicalAgentHarness
    tasks: set[asyncio.Task]


def create_runtime(services: ServiceContainer | None = None) -> ApiRuntime:
    services = services or create_service_container()
    guard = EgressGuard(services.repository)
    registry = create_tool_registry(services)
    return ApiRuntime(
        services=services, guard=guard,
        harness=ClinicalAgentHarness(services, registry, guard),
        tasks=set(),
    )


runtime = create_runtime()


async def get_runtime() -> ApiRuntime:
    return runtime


Runtime = Annotated[ApiRuntime, Depends(get_runtime)]
