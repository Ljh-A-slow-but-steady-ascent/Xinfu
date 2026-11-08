"""Small, bounded native tool-calling client for the local evidence agent."""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

import requests

from backend.app.services.privacy import EgressGuard, EgressRequest


class RagLLMError(RuntimeError):
    """A safe-to-display failure; provider responses and credentials stay private."""


def _endpoint(value: str, default: str) -> str:
    base = (value or default).rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _json_content(value: Any) -> tuple[Any, bool]:
    """Inspect structured tool results even when transported as JSON strings."""
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value), True
        except (ValueError, TypeError):
            pass
    return value, False


class RagLLM:
    """Use the configured chat provider without retries or a second agent stack.

    ``step`` returns the provider's assistant message, including native function
    calls with JSON-string arguments. The controller owns tool execution,
    citation validation, stopping conditions, and the overall request budget.
    """

    def __init__(self, guard: EgressGuard | None = None, model_id: str | None = None) -> None:
        self.guard = guard or EgressGuard()
        self.last_error: str | None = None
        self.provider = os.getenv("DERMAGUARD_CHAT_PROVIDER", "mock").strip().lower()
        self.model = model_id or os.getenv("DERMAGUARD_RAG_MODEL", "auto").strip() or "auto"
        self._api_key = ""
        self._url = ""
        # Offline mode is authoritative, including when a model was explicit.
        if self.provider == "mock":
            return
        selected = self.model
        if selected == "auto":
            if self.provider == "deepseek":
                selected = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            elif self.provider == "groq":
                selected = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            elif self.provider in {"zhipu", "zhipu_glm", "glm"}:
                selected = os.getenv("ZHIPUAI_MODEL", "glm-5.3-flash")
            else:
                return
        if selected.startswith("deepseek-"):
            self.provider = "deepseek"
            prefix, default = "DEEPSEEK", "https://api.deepseek.com"
        elif selected.startswith("glm-"):
            self.provider = "zhipu"
            prefix, default = "ZHIPUAI", "https://open.bigmodel.cn/api/paas/v4"
        elif selected.startswith(("openai/gpt-oss-", "qwen/")) or self.provider == "groq":
            self.provider = "groq"
            prefix, default = "GROQ", "https://api.groq.com/openai/v1"
        else:
            return
        self.model = selected
        self._api_key = os.getenv(f"{prefix}_API_KEY", "")
        self._url = _endpoint(os.getenv(f"{prefix}_BASE_URL", ""), default)

    @property
    def available(self) -> bool:
        return bool(self._api_key and self._url)

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        safe_messages = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise RagLLMError("RAG 模型消息格式无效")
            # Only protocol routing metadata bypasses payload inspection. A
            # function's `name` is not a patient's prohibited `name` field.
            content, was_json = _json_content(message.get("content"))
            if not isinstance(message.get("content"), (str, type(None))):
                raise RagLLMError("RAG 模型只接收脱敏文本")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise RagLLMError("RAG 工具消息格式无效")
            arguments = []
            argument_is_json = []
            for call in calls:
                try:
                    argument = call["function"]["arguments"]
                    if not isinstance(argument, str):
                        raise TypeError
                    parsed, is_json = _json_content(argument)
                    arguments.append(parsed)
                    argument_is_json.append(is_json)
                except (KeyError, TypeError) as exc:
                    raise RagLLMError("RAG 工具参数格式无效") from exc
            checked = self.guard.enforce(EgressRequest(
                channel="llm", destination=self.model, tool_name="agentic_rag",
                payload={"content": content, "arguments": arguments},
            ))
            safe = checked.sanitized_payload
            safe_message = {key: copy.deepcopy(message[key]) for key in ("role", "tool_call_id", "tool_calls") if key in message}
            safe_message["content"] = json.dumps(safe["content"], ensure_ascii=False) if was_json else safe["content"]
            for index, call in enumerate(safe_message.get("tool_calls", [])):
                argument = safe["arguments"][index]
                call["function"]["arguments"] = json.dumps(argument, ensure_ascii=False) if argument_is_json[index] else argument
            safe_messages.append(safe_message)
        return safe_messages

    def step(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float | None = None,
    ) -> dict[str, Any]:
        self.last_error = None
        if not self.available:
            raise RagLLMError("RAG 工具调用模型未配置或处于离线模式")
        try:
            configured_budget = float(os.getenv("DERMAGUARD_RAG_LLM_TIMEOUT", "12"))
            remaining_budget = float(timeout) if timeout is not None else configured_budget
            if any(not math.isfinite(value) or value <= 0 for value in (configured_budget, remaining_budget)):
                raise ValueError
            budget = min(configured_budget, remaining_budget, 30.0)
        except (TypeError, ValueError) as exc:
            raise RagLLMError("RAG 模型超时配置无效") from exc
        try:
            safe_messages = self._messages(messages)
            body: dict[str, Any] = {
                "model": self.model, "stream": False, "temperature": 0.05,
                "max_tokens": 1600, "messages": safe_messages,
            }
            if tools:
                body.update({"tools": tools, "tool_choice": "auto"})
            if self.provider in {"deepseek", "zhipu"}:
                body["thinking"] = {"type": "disabled"}
            if self.provider == "groq" and self.model.startswith("openai/gpt-oss-"):
                body["reasoning_effort"] = "low"
            response = requests.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body, timeout=budget,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError
            content = message.get("content")
            calls = message.get("tool_calls") or []
            if not isinstance(content, (str, type(None))) or not isinstance(calls, list):
                raise TypeError
            for call in calls:
                if (not isinstance(call, dict) or not isinstance(call.get("id"), str)
                    or call.get("type") != "function" or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("name"), str)
                    or not isinstance(call["function"].get("arguments"), str)):
                    raise TypeError
            if not calls and not (content and content.strip()):
                raise ValueError
            result: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                result["tool_calls"] = calls
            return result
        except Exception as exc:
            # Provider exceptions can contain URLs, bearer credentials, or raw
            # content. Only expose a stable category to callers and traces.
            if isinstance(exc, RagLLMError):
                self.last_error = str(exc)
                raise
            if isinstance(exc, requests.Timeout):
                message = "RAG 模型响应超时"
            elif isinstance(exc, requests.RequestException):
                message = "RAG 模型服务暂时不可用"
            else:
                message = "RAG 模型响应无效或出站检查未通过"
            self.last_error = message
            raise RagLLMError(message) from None
