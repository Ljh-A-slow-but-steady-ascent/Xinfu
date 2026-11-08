from __future__ import annotations

import json
from typing import Any


class MockChatProvider:
    """Offline planner; callers retain deterministic local parsing defaults."""

    name = "mock"
    model = "local-mock"

    def plan(self, payload: dict[str, Any], system_prompt: str) -> str:
        return json.dumps({}, ensure_ascii=False)
