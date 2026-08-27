"""Request preparation for the visual agent."""

from __future__ import annotations

import copy
from typing import Any, Optional

from ..api_models import ChatCompletionRequest
from .config import VisualProxySettings
from .evidence import ImageRegistry
from .images import image_url_for_visual_provider


def request_has_images(request: ChatCompletionRequest) -> bool:
    """Return whether any conversation message contains an image payload."""

    return any(
        getattr(part, "type", None) in {"image", "image_url"}
        and getattr(part, "image_url", None) is not None
        for message in request.messages
        if isinstance(message.content, list)
        for part in message.content
    )


def should_use_visual_proxy(
    visual_remote_url: Optional[str], request: ChatCompletionRequest
) -> bool:
    """Keep the native path untouched when the proxy is disabled."""

    return bool(visual_remote_url and visual_remote_url.strip()) and request_has_images(
        request
    )


def requested_enable_thinking(request: ChatCompletionRequest) -> Optional[bool]:
    controls: list[tuple[str, bool]] = []
    template_kwargs = request.chat_template_kwargs or {}
    for name in ("enable_thinking", "thinking"):
        value = template_kwargs.get(name)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"chat_template_kwargs.{name} must be a boolean")
        controls.append((f"chat_template_kwargs.{name}", value))

    if request.reasoning_effort is not None:
        controls.append(
            (
                "reasoning_effort",
                request.reasoning_effort not in {"none", "off", "disabled"},
            )
        )
    if controls and any(value != controls[0][1] for _, value in controls[1:]):
        names = ", ".join(name for name, _ in controls)
        raise ValueError(
            f"Conflicting thinking controls: {names} must express the same boolean value"
        )
    return controls[0][1] if controls else None


def apply_visual_thinking_policy(
    request: ChatCompletionRequest, settings: VisualProxySettings
) -> ChatCompletionRequest:
    """Apply the proxy thinking policy without mutating the caller's request."""

    requested = requested_enable_thinking(request)
    if settings.thinking_policy == "force_on":
        enabled = True
    elif settings.thinking_policy == "force_off":
        enabled = False
    elif settings.thinking_policy == "request":
        enabled = requested if requested is not None else True
    else:
        raise ValueError(
            f"Unsupported visual thinking policy: {settings.thinking_policy!r}"
        )

    template_kwargs = copy.deepcopy(request.chat_template_kwargs or {})
    template_kwargs.update(enable_thinking=enabled, thinking=enabled)
    reasoning_effort = request.reasoning_effort
    if enabled and reasoning_effort in {None, "none", "off", "disabled"}:
        reasoning_effort = "high"
    elif not enabled:
        reasoning_effort = "none"
    return request.model_copy(
        update={
            "chat_template_kwargs": template_kwargs,
            "reasoning_effort": reasoning_effort,
        }
    )


def resolve_public_reasoning_enabled(
    requested: Optional[bool], settings: VisualProxySettings
) -> bool:
    """Apply the public-output gate without overriding an explicit opt-out."""

    return requested is not False and settings.thinking_policy != "force_off"


def _image_source(part: dict[str, Any]) -> Optional[str]:
    if part.get("type") not in {"image", "image_url"}:
        return None
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if image_url is None and isinstance(part.get("image"), str):
        image_url = part["image"]
    if not isinstance(image_url, str) or not image_url:
        raise ValueError(
            "OpenAI image content must contain a non-empty image_url.url value"
        )
    return image_url


def _assistant_closes_user_turn(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return False
    blocks = message.get("anthropic_content_blocks")
    return not (
        isinstance(blocks, list)
        and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in blocks
        )
    )


def replace_images_with_tags(
    messages: list[dict[str, Any]],
    registry: ImageRegistry,
    settings: Optional[VisualProxySettings] = None,
) -> list[dict[str, Any]]:
    """Copy messages while replacing image parts with stable XML tags."""

    last_completed_turn = max(
        (
            index
            for index, message in enumerate(messages)
            if _assistant_closes_user_turn(message)
        ),
        default=-1,
    )
    current_user_index = next(
        (
            index
            for index in range(last_completed_turn + 1, len(messages))
            if messages[index].get("role") == "user"
        ),
        None,
    )

    transformed = copy.deepcopy(messages)
    for message_index, message in enumerate(transformed):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(f"Unsupported OpenAI message content item: {part!r}")
            source = _image_source(part)
            if source is None:
                new_content.append(part)
                continue
            if settings is not None:
                source = image_url_for_visual_provider(source, settings)
            byte_size = 0
            if source.startswith("data:"):
                byte_size = (len(source.split(",", 1)[1]) * 3) // 4
            origin = "tool_response" if message.get("role") == "tool" else "user"
            tag = registry.add(
                source=source,
                origin=origin,
                byte_size=byte_size,
                current_user_turn=(
                    current_user_index is not None
                    and message_index >= current_user_index
                    and message.get("role") in {"user", "tool"}
                ),
            )
            new_content.append({"type": "text", "text": tag})
        message["content"] = new_content
    return transformed
