"""Opt-in, per-request trajectory dumps for the visual chat proxy."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

TRACE_DUMP_ENV = "LIGHTLLM_VISUAL_TRACE_DUMP"
TRACE_DUMP_DIR_ENV = "LIGHTLLM_VISUAL_TRACE_DUMP_DIR"
DEFAULT_TRACE_DUMP_DIR = "/tmp/lightllm_visual_trajectories"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def visual_trace_dump_dir_from_env() -> Optional[Path]:
    raw_enabled = os.getenv(TRACE_DUMP_ENV, "0").strip().lower()
    if raw_enabled in {"", "0", "false", "no", "off"}:
        return None
    if raw_enabled not in {"1", "true", "yes", "on"}:
        raise ValueError(f"{TRACE_DUMP_ENV} must be one of 1/true/yes/on or 0/false/no/off")
    raw_dir = os.getenv(TRACE_DUMP_DIR_ENV, DEFAULT_TRACE_DUMP_DIR).strip()
    if not raw_dir:
        raise ValueError(f"{TRACE_DUMP_DIR_ENV} must not be empty when tracing is enabled")
    directory = Path(raw_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"{TRACE_DUMP_DIR_ENV} must point to a directory: {directory}")
    return directory


class VisualTraceRecorder:
    """Collect one complete external-request trajectory in one JSON file."""

    def __init__(self, trace_id: str, directory: Optional[Path], request: dict[str, Any]):
        self.trace_id = trace_id
        self.directory = directory
        self._started_monotonic = time.monotonic()
        self._filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{trace_id}.json"
        if directory is None:
            self._record = {}
            return
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": trace_id,
            "started_at": _utc_now(),
            "status": "running",
            "request": copy.deepcopy(request),
            "events": [],
        }

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def event(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        events = self._record["events"]
        events.append(
            {
                "index": len(events) + 1,
                "type": kind,
                "timestamp": _utc_now(),
                "elapsed_ms": round((time.monotonic() - self._started_monotonic) * 1000, 3),
                **copy.deepcopy(data),
            }
        )

    def finish_success(self, response: Any) -> None:
        if not self.enabled:
            return
        self._record["status"] = "success"
        self._record["response"] = copy.deepcopy(response)
        self._finish()

    def finish_error(self, exc: BaseException, traceback_text: str) -> None:
        if not self.enabled:
            return
        self._record["status"] = "error"
        self._record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback_text,
        }
        self._finish()

    def _finish(self) -> None:
        self._record["finished_at"] = _utc_now()
        self._record["duration_ms"] = round((time.monotonic() - self._started_monotonic) * 1000, 3)

    async def flush(self) -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._write_json)
        except Exception:
            logger.exception(
                "[visual-chat-proxy][trace_dump_error] trace_id=%s dir=%s",
                self.trace_id,
                self.directory,
            )

    def _write_json(self) -> None:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        final_path = self.directory / self._filename
        temporary_path = self.directory / (f".{self._filename}.tmp-{os.getpid()}-{id(self)}")
        payload = json.dumps(self._record, ensure_ascii=False, indent=2)
        fd = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, final_path)
            logger.info(
                "[visual-chat-proxy][trace_dump_written] trace_id=%s path=%s",
                self.trace_id,
                final_path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


_main_model_trace: contextvars.ContextVar[Optional[tuple[VisualTraceRecorder, str, int]]] = contextvars.ContextVar(
    "lightllm_visual_main_model_trace", default=None
)


@contextmanager
def bind_main_model_trace(recorder: VisualTraceRecorder, choice_trace_id: str, step: int) -> Iterator[None]:
    token = _main_model_trace.set((recorder, choice_trace_id, step))
    try:
        yield
    finally:
        _main_model_trace.reset(token)


def record_rendered_main_prompt(prompt: str) -> None:
    """Record the exact post-chat-template prompt while inside a visual agent turn."""

    current = _main_model_trace.get()
    if current is None:
        return
    recorder, choice_trace_id, step = current
    recorder.event(
        "main_model_rendered_prompt",
        choice_trace_id=choice_trace_id,
        step=step,
        prompt=prompt,
    )
