"""One resident image worker shared by case analysis and candidate exploration."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODEL_PYTHON = Path(os.getenv("DERMAGUARD_VISION_PYTHON") or Path(sys.prefix).parent / "dermfm-zero/bin/python")
WORKER = Path(__file__).with_name("frozen_dermfm_worker.py")
PREFIX = b"DERMAGUARD_RESULT="


class VisionWorker:
    def __init__(self, command: list[str] | None = None, capacity: int = 4) -> None:
        self.command = command or [str(MODEL_PYTHON), str(WORKER), "--serve"]
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(capacity)

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.process = subprocess.Popen(
            self.command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, bufsize=0,
        )
        assert self.process.stdin and self.process.stdout
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def predict(self, image_path: Path | None, metadata: dict[str, Any], *,
                mode: str = "image_text", top_k: int = 3, timeout: float = 300) -> dict:
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("图片分析队列已满，请稍后重试")
        try:
            if not self.lock.acquire(timeout=timeout):
                raise TimeoutError("图片分析排队超时，请稍后重试")
            try:
                return self._request({"image": str(image_path) if image_path else None,
                                      "metadata": metadata, "mode": mode, "top_k": top_k}, timeout)
            finally:
                self.lock.release()
        finally:
            self.slots.release()

    def _request(self, payload: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        try:
            self._start()
            assert self.process and self.process.stdin and self.process.stdout
            message = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            sent = 0
            while sent < len(message):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError("图片分析响应超时")
                try:
                    sent += os.write(self.process.stdin.fileno(), message[sent:])
                except BlockingIOError:
                    continue
            response = bytearray()
            while True:
                while b"\n" in response:
                    line, _, rest = response.partition(b"\n")
                    response = bytearray(rest)
                    if line.startswith(PREFIX):
                        result = json.loads(line[len(PREFIX):])
                        if result.get("error"):
                            raise ValueError(str(result["error"]))
                        return result
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError("图片分析响应超时")
                try:
                    block = os.read(self.process.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not block:
                    raise RuntimeError("图片分析进程已退出")
                response.extend(block)
                if len(response) > 2_000_000:
                    raise RuntimeError("图片分析返回内容超出限制")
        except Exception:
            self.close()
            raise

    def warmup(self) -> dict:
        with self.lock:
            return self._request({"action": "warmup"}, 300)


worker = VisionWorker()
atexit.register(worker.close)
