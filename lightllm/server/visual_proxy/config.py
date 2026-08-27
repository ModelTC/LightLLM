"""Configuration and error types for the visual proxy package."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .constants import BUILTIN_TRACE_FORMATS, THINKING_POLICIES
from .images import normalize_visual_remote_url
from ..visual_trace import visual_trace_dump_dir_from_env


class VisualChatProxyError(RuntimeError):
    """Raised when the enabled proxy cannot complete its internal agent loop."""


class VisualProxyUpstreamError(VisualChatProxyError):
    """Raised when the configured visual upstream returns an unusable response."""


class VisualProxyUpstreamRejectedError(VisualProxyUpstreamError):
    """Raised for a non-retryable upstream 4xx without opening the circuit."""


class VisualProxyTimeoutError(VisualChatProxyError):
    """Raised when the visual upstream or the complete agent loop times out."""


class VisualProxyCapacityError(VisualChatProxyError):
    """Raised when the bounded visual upstream queue is saturated."""


@dataclass(frozen=True)
class VisualProxySettings:
    remote_url: str
    builtin_trace_format: str = "natural"
    thinking_policy: str = "request"
    empty_output_retries: int = 2
    trace_dump_dir: Optional[Path] = None
    remote_model: Optional[str] = None
    remote_api_key: Optional[str] = None
    remote_headers: tuple[tuple[str, str], ...] = ()
    allow_insecure_remote_url: bool = False
    remote_timeout: float = 180.0
    remote_connect_timeout: float = 10.0
    remote_max_retries: int = 2
    remote_max_concurrency: int = 64
    remote_queue_timeout: float = 10.0
    max_inflight_requests: int = 64
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0
    agent_timeout: float = 600.0
    max_images: int = 8
    max_image_bytes: int = 20 * 1024 * 1024
    max_total_image_bytes: int = 40 * 1024 * 1024
    max_remote_response_bytes: int = 64 * 1024
    max_upstream_body_bytes: int = 1024 * 1024
    max_choices: int = 4
    allow_local_files: bool = False
    local_file_roots: tuple[Path, ...] = ()
    allow_remote_image_urls: bool = False
    allow_http_image_urls: bool = False
    allow_private_remote_image_urls: bool = False
    remote_image_hosts: tuple[str, ...] = ()

    @classmethod
    def from_args(cls, args: Any) -> "VisualProxySettings":
        allow_insecure_remote_url = bool(
            getattr(args, "visual_allow_insecure_remote_url", False)
        )
        remote_url = normalize_visual_remote_url(
            str(getattr(args, "visual_remote_url", "") or ""),
            allow_insecure_http=allow_insecure_remote_url,
        )

        api_key_env = str(
            getattr(args, "visual_remote_api_key_env", "LIGHTLLM_VISUAL_REMOTE_API_KEY")
        )
        remote_api_key = os.environ.get(api_key_env) or None
        headers_env = str(
            getattr(args, "visual_remote_headers_env", "LIGHTLLM_VISUAL_REMOTE_HEADERS")
        )
        remote_headers: tuple[tuple[str, str], ...] = ()
        raw_headers = os.environ.get(headers_env)
        if raw_headers:
            try:
                parsed_headers = json.loads(raw_headers)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{headers_env} must be a JSON object of HTTP headers"
                ) from exc
            if not isinstance(parsed_headers, dict):
                raise ValueError(f"{headers_env} must be a JSON object of HTTP headers")
            normalized_headers = []
            for name, value in parsed_headers.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise ValueError(
                        f"{headers_env} header names and values must be strings"
                    )
                if "\n" in name or "\r" in name or "\n" in value or "\r" in value:
                    raise ValueError(f"{headers_env} contains an invalid HTTP header")
                if name.lower() in {"host", "content-length"}:
                    raise ValueError(f"{headers_env} must not override {name}")
                normalized_headers.append((name, value))
            remote_headers = tuple(normalized_headers)

        allow_local_files = bool(getattr(args, "visual_allow_local_files", False))
        raw_roots = tuple(getattr(args, "visual_local_file_root", None) or ())
        local_file_roots: tuple[Path, ...] = ()
        if allow_local_files:
            if not raw_roots:
                raise ValueError(
                    "--visual_allow_local_files requires at least one "
                    "--visual_local_file_root"
                )
            resolved_roots = []
            for raw_root in raw_roots:
                root = Path(raw_root).expanduser().resolve(strict=True)
                if not root.is_dir():
                    raise ValueError(
                        f"Visual local file root is not a directory: {root}"
                    )
                resolved_roots.append(root)
            local_file_roots = tuple(resolved_roots)
        elif raw_roots:
            raise ValueError(
                "--visual_local_file_root has no effect unless "
                "--visual_allow_local_files is set"
            )

        settings = cls(
            remote_url=remote_url,
            builtin_trace_format=str(
                getattr(args, "visual_builtin_trace_format", "natural")
            ),
            thinking_policy=str(
                getattr(args, "visual_thinking_policy", None)
                or os.getenv("THINKING_POLICY", "request")
            ),
            empty_output_retries=int(
                os.getenv("EMPTY_OUTPUT_RETRIES", "2")
                if getattr(args, "visual_empty_output_retries", None) is None
                else getattr(args, "visual_empty_output_retries")
            ),
            trace_dump_dir=visual_trace_dump_dir_from_env(),
            remote_model=getattr(args, "visual_remote_model", None),
            remote_api_key=remote_api_key,
            remote_headers=remote_headers,
            allow_insecure_remote_url=allow_insecure_remote_url,
            remote_timeout=float(getattr(args, "visual_remote_timeout", 180.0)),
            remote_connect_timeout=float(
                getattr(args, "visual_remote_connect_timeout", 10.0)
            ),
            remote_max_retries=int(getattr(args, "visual_remote_max_retries", 2)),
            remote_max_concurrency=int(
                getattr(args, "visual_remote_max_concurrency", 64)
            ),
            remote_queue_timeout=float(
                getattr(args, "visual_remote_queue_timeout", 10.0)
            ),
            max_inflight_requests=int(
                getattr(args, "visual_max_inflight_requests", 64)
            ),
            circuit_failure_threshold=int(
                getattr(args, "visual_circuit_failure_threshold", 5)
            ),
            circuit_recovery_seconds=float(
                getattr(args, "visual_circuit_recovery_seconds", 30.0)
            ),
            agent_timeout=float(getattr(args, "visual_agent_timeout", 600.0)),
            max_images=int(getattr(args, "visual_max_images", 8)),
            max_image_bytes=int(
                getattr(args, "visual_max_image_bytes", 20 * 1024 * 1024)
            ),
            max_total_image_bytes=int(
                getattr(args, "visual_max_total_image_bytes", 40 * 1024 * 1024)
            ),
            max_remote_response_bytes=int(
                getattr(args, "visual_max_remote_response_bytes", 64 * 1024)
            ),
            max_upstream_body_bytes=int(
                getattr(args, "visual_max_upstream_body_bytes", 1024 * 1024)
            ),
            max_choices=int(getattr(args, "visual_max_choices", 4)),
            allow_local_files=allow_local_files,
            local_file_roots=local_file_roots,
            allow_remote_image_urls=bool(
                getattr(args, "visual_allow_remote_image_urls", False)
            ),
            allow_http_image_urls=bool(
                getattr(args, "visual_allow_http_image_urls", False)
            ),
            allow_private_remote_image_urls=bool(
                getattr(args, "visual_allow_private_remote_image_urls", False)
            ),
            remote_image_hosts=tuple(
                str(host).lower().rstrip(".")
                for host in (getattr(args, "visual_remote_image_host", None) or ())
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        normalize_visual_remote_url(
            self.remote_url,
            allow_insecure_http=self.allow_insecure_remote_url,
        )
        if self.builtin_trace_format not in BUILTIN_TRACE_FORMATS:
            raise ValueError(
                "--builtin-trace-format/--visual_builtin_trace_format must be 'xml' or 'natural'"
            )
        if self.thinking_policy not in THINKING_POLICIES:
            raise ValueError(
                "--visual_thinking_policy/THINKING_POLICY must be 'request', 'force_on', or 'force_off'"
            )
        if self.empty_output_retries < 0:
            raise ValueError("--visual_empty_output_retries must be non-negative")
        positive_values = {
            "visual_remote_timeout": self.remote_timeout,
            "visual_remote_connect_timeout": self.remote_connect_timeout,
            "visual_remote_max_concurrency": self.remote_max_concurrency,
            "visual_remote_queue_timeout": self.remote_queue_timeout,
            "visual_max_inflight_requests": self.max_inflight_requests,
            "visual_circuit_failure_threshold": self.circuit_failure_threshold,
            "visual_circuit_recovery_seconds": self.circuit_recovery_seconds,
            "visual_agent_timeout": self.agent_timeout,
            "visual_max_images": self.max_images,
            "visual_max_image_bytes": self.max_image_bytes,
            "visual_max_total_image_bytes": self.max_total_image_bytes,
            "visual_max_remote_response_bytes": self.max_remote_response_bytes,
            "visual_max_upstream_body_bytes": self.max_upstream_body_bytes,
            "visual_max_choices": self.max_choices,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"--{name} must be greater than zero")
        if self.remote_max_retries < 0 or self.remote_max_retries > 10:
            raise ValueError("--visual_remote_max_retries must be between 0 and 10")
        if self.allow_http_image_urls and not self.allow_remote_image_urls:
            raise ValueError(
                "--visual_allow_http_image_urls requires "
                "--visual_allow_remote_image_urls"
            )
        if self.allow_private_remote_image_urls and not self.allow_remote_image_urls:
            raise ValueError(
                "--visual_allow_private_remote_image_urls requires "
                "--visual_allow_remote_image_urls"
            )
        if self.allow_remote_image_urls and not self.remote_image_hosts:
            raise ValueError(
                "--visual_allow_remote_image_urls requires at least one "
                "--visual_remote_image_host"
            )
        if not self.allow_remote_image_urls and self.remote_image_hosts:
            raise ValueError(
                "--visual_remote_image_host requires --visual_allow_remote_image_urls"
            )
        for host in self.remote_image_hosts:
            if not host or "/" in host or ":" in host:
                raise ValueError(
                    "--visual_remote_image_host values must be exact DNS hostnames"
                )


def validate_visual_proxy_startup(args: Any) -> None:
    """Fail fast before model loading when production proxy settings are unsafe."""

    if getattr(args, "visual_remote_url", None):
        VisualProxySettings.from_args(args)
