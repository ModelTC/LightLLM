"""Image registration and evidence accounting for the visual chat proxy."""

from __future__ import annotations

import base64
import copy
import hashlib
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .constants import (
    IMAGE_ALIAS_PATTERN,
    IMAGE_TAG_PATTERN,
    INVALID_BUILTIN_VISION_RESULT_MARKERS,
    TRUNCATED_FINISH_REASONS,
)


def _valid_builtin_vision_result(value: str) -> bool:
    text = value.strip()
    return bool(text) and not any(
        marker in text for marker in INVALID_BUILTIN_VISION_RESULT_MARKERS
    )


@dataclass(frozen=True)
class RegisteredImage:
    tag: str
    source: str
    origin: str
    current_user_turn: bool = False


class ImageRegistry:
    """Assign stable conversation-order XML tags to OpenAI image content."""

    def __init__(
        self,
        max_images: int = 8,
        max_total_image_bytes: int = 40 * 1024 * 1024,
    ) -> None:
        self._images: list[RegisteredImage] = []
        self._max_images = max_images
        self._max_total_image_bytes = max_total_image_bytes
        self._total_image_bytes = 0
        self._current_user_turn_tags: list[str] = []
        self._digest_by_tag: dict[str, str] = {}

    def add(
        self,
        source: str,
        origin: str,
        byte_size: int = 0,
        *,
        current_user_turn: bool = False,
    ) -> str:
        if self._max_images > 0 and len(self._images) >= self._max_images:
            raise ValueError(
                f"Visual proxy accepts at most {self._max_images} images per request"
            )

        if (
            self._max_total_image_bytes > 0
            and self._total_image_bytes + byte_size > self._max_total_image_bytes
        ):
            raise ValueError(
                "Visual proxy image payloads exceed the "
                f"{self._max_total_image_bytes}-byte request limit"
            )
        tag = f"<image_{len(self._images) + 1}/>"
        self._images.append(
            RegisteredImage(
                tag=tag,
                source=source,
                origin=origin,
                current_user_turn=current_user_turn,
            )
        )
        if current_user_turn:
            self._current_user_turn_tags.append(tag)
        if source.startswith("data:"):
            with suppress(ValueError, TypeError):
                payload = source.split(",", 1)[1]
                self._digest_by_tag[tag] = hashlib.sha256(
                    base64.b64decode(payload, validate=True)
                ).hexdigest()
        self._total_image_bytes += byte_size
        return tag

    def resolve(self, label: str) -> RegisteredImage:
        match = IMAGE_TAG_PATTERN.search(str(label))
        if match is None:
            match = IMAGE_ALIAS_PATTERN.fullmatch(str(label).strip().rstrip(":"))
        if match is None:
            raise ValueError(
                f"vision_reader image must be an XML tag such as <image_1/>; received {label!r}. "
                f"Available tags: {self.available_tags()}"
            )
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(self._images):
            raise ValueError(
                f"Unknown vision_reader image tag {label!r}. Available tags: {self.available_tags()}"
            )
        return self._images[index]

    def available_tags(self) -> str:
        return ", ".join(image.tag for image in self._images) or "(none)"

    def tags(self) -> list[str]:
        return [image.tag for image in self._images]

    def current_user_turn_tags(self) -> list[str]:
        return list(self._current_user_turn_tags)

    def bind_digest(self, tag: str, digest_sha256: str) -> None:
        if digest_sha256:
            self.resolve(tag)
            self._digest_by_tag[tag] = digest_sha256

    def digest_for_tag(self, tag: str) -> str:
        return self._digest_by_tag.get(self.resolve(tag).tag, "")

    def __len__(self) -> int:
        return len(self._images)


class EvidenceLedger:
    """Record visual execution outcomes for observability, never as an output gate."""

    def __init__(self, registry: ImageRegistry) -> None:
        self.registry = registry
        current_turn_tags = registry.current_user_turn_tags()
        # LightLLM does not persist Nova's private continuation cache. When a
        # follow-up refers to historical images, re-read them rather than
        # trusting caller-replayed public reasoning as evidence.
        self.target_tags = current_turn_tags or registry.tags()
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        image_tag: str,
        result: str,
        step: int,
        call_index: int,
        tool_call_id: str,
        task: str,
        finish_reason: str = "",
        image_digest_sha256: str = "",
        terminal_failed: str = "",
    ) -> bool:
        if image_digest_sha256:
            self.registry.bind_digest(image_tag, image_digest_sha256)
        accepted = (
            _valid_builtin_vision_result(result)
            and finish_reason not in TRUNCATED_FINISH_REASONS
            and not terminal_failed
        )
        self.records.append(
            {
                "agent_step_index": step,
                "vision_call_index": call_index,
                "image_tag": image_tag,
                "content_digest_sha256": self.registry.digest_for_tag(image_tag),
                "content_digest_verified": bool(
                    self.registry.digest_for_tag(image_tag)
                ),
                "tool_call_id": tool_call_id,
                "task_digest_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
                "provider_finish_reason": finish_reason,
                "terminal_failed": terminal_failed,
                "accepted_as_evidence": accepted,
            }
        )
        return accepted

    def covered_tags(self) -> list[str]:
        covered = {
            record["image_tag"]
            for record in self.records
            if record.get("accepted_as_evidence") is True
        }
        return [tag for tag in self.registry.tags() if tag in covered]

    def missing_tags(self, *, visual_answer_required: bool) -> list[str]:
        if not visual_answer_required:
            return []
        covered = set(self.covered_tags())
        return [tag for tag in self.target_tags if tag not in covered]

    def snapshot(self, *, visual_answer_required: bool) -> dict[str, Any]:
        missing = self.missing_tags(visual_answer_required=visual_answer_required)
        return {
            "schema_version": "lightllm.visual_proxy.image_evidence.v1",
            "policy": "observational_agent_decides_vision",
            "current_user_turn_image_tags": list(self.target_tags),
            "vision_results": copy.deepcopy(self.records),
            "covered_image_tags": self.covered_tags(),
            "missing_image_tags": missing,
            "coverage_complete": not missing,
        }
