from __future__ import annotations

import base64
import io
import uuid
from typing import Any

import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

from lightllm.server.actionserver.objs import (
    ActionRequest,
    PrefixContextIdentity,
    PrefixContextOp,
    PrefixContextRef,
)
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.inference_runtime.output_plan import OutputPlan
from lightllm.utils.envs_utils import get_unique_server_name


_FLAT_CONTEXT_FIELDS = frozenset(
    {
        "context_id",
        "context_version",
        "server_epoch",
        "persist_prefix",
        "replace_prefix",
    }
)
_FLAT_IDENTITY_FIELDS = frozenset(
    {
        "context_id",
        "context_version",
        "server_epoch",
    }
)


class _ContextOwnerJSONResponse(JSONResponse):
    """Resolve a provisional context only after ASGI sends the response body."""

    def __init__(
        self,
        content,
        *,
        httpserver_manager,
        context_owner_identity,
        status_code: int = 200,
    ) -> None:
        super().__init__(content, status_code=status_code)
        self._httpserver_manager = httpserver_manager
        self._context_owner_identity = context_owner_identity
        self._owner_resolved = False

    def _resolve_owner(self, *, delivered: bool) -> None:
        if self._owner_resolved:
            return
        if not self._httpserver_manager.resolve_action_context_owner(
            self._context_owner_identity,
            delivered=delivered,
        ):
            raise RuntimeError(
                "persistent context owner handshake was lost during response send"
            )
        self._owner_resolved = True

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            self._resolve_owner(delivered=False)
            raise
        else:
            self._resolve_owner(delivered=True)


def _context_flag(body: dict[str, Any], name: str) -> bool:
    if name not in body:
        return False
    value = body[name]
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _json_context_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("context_version must be a positive JSON integer")
    return value


def _parse_close_context_version(value: Any) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise ValueError("context_version must be a positive decimal integer")
    version = int(value, 10)
    if version <= 0:
        raise ValueError("context_version must be a positive decimal integer")
    return version


def _image_to_item(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return {"type": "url", "data": value}
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        # Validate here so malformed payloads fail before shared-cache allocation.
        base64.b64decode(encoded, validate=True)
        return {"type": "base64", "data": encoded}

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError("each image must be base64, URL, or an HxWx3 array")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.rint(array)
    if array.size and (array.min() < 0 or array.max() > 255):
        raise ValueError("image arrays must contain values in [0, 255]")
    array = array.astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=96)
    return {
        "type": "base64",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _ordered_images(images: Any, image_keys: tuple[str, ...]) -> list[Any]:
    if isinstance(images, dict):
        priority = {}
        for index, key in enumerate(image_keys):
            priority[key] = index
            priority[key.rsplit(".", 1)[-1]] = index
        insertion_order = {key: index for index, key in enumerate(images)}
        keys = sorted(
            images,
            key=lambda key: (priority.get(key, len(priority)), insertion_order[key]),
        )
        return [images[key] for key in keys]
    if isinstance(images, list):
        try:
            array = np.asarray(images)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.dtype != object and array.ndim == 3:
            return [images]
        return images
    return [images]


def _resolve_prefix_context(body: dict[str, Any]) -> PrefixContextRef:
    if not isinstance(body, dict):
        raise ValueError("action request body must be an object")

    if "prefix_context" in body:
        conflicting = _FLAT_CONTEXT_FIELDS.intersection(body)
        if conflicting:
            raise ValueError(
                "prefix_context cannot be combined with flat context fields: " + ", ".join(sorted(conflicting))
            )
        return PrefixContextRef.from_dict(body["prefix_context"])

    persist_prefix = _context_flag(body, "persist_prefix")
    replace_prefix = _context_flag(body, "replace_prefix")
    if persist_prefix and replace_prefix:
        raise ValueError("persist_prefix and replace_prefix cannot both be true")

    if persist_prefix:
        mixed_identity_fields = {"context_version", "server_epoch"}.intersection(body)
        if mixed_identity_fields:
            raise ValueError(
                "context create must not include existing-identity fields: "
                + ", ".join(sorted(mixed_identity_fields))
            )
        return PrefixContextRef(
            op=PrefixContextOp.CREATE,
            context_id=(body["context_id"] if "context_id" in body else uuid.uuid4().hex),
        )

    present_identity_fields = _FLAT_IDENTITY_FIELDS.intersection(body)
    if not present_identity_fields and not replace_prefix:
        return PrefixContextRef()

    op = PrefixContextOp.REPLACE if replace_prefix else PrefixContextOp.REUSE
    if present_identity_fields != _FLAT_IDENTITY_FIELDS:
        missing = _FLAT_IDENTITY_FIELDS.difference(present_identity_fields)
        raise ValueError(
            f"context {op.value} requires exact identity fields; missing: "
            + ", ".join(sorted(missing))
        )

    identity = PrefixContextIdentity(
        context_id=body["context_id"],
        version=_json_context_version(body["context_version"]),
        server_epoch=body["server_epoch"],
    )
    return PrefixContextRef(op, identity)


async def generate_actions(request: Request, httpserver_manager) -> JSONResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("action request body must be an object")
    config = httpserver_manager.vla_config
    model_type = body.get("model_type")
    if model_type is not None and model_type != config.model_type.value:
        raise ValueError(f"request model_type={model_type} does not match {config.model_type.value}")
    prefix_context = _resolve_prefix_context(body)
    if prefix_context.op is PrefixContextOp.CLOSE:
        raise ValueError("use DELETE /v1/vla/contexts/{context_id} to close a context")
    if prefix_context.op.requires_prefix_inputs:
        if body.get("prompt") is None:
            raise ValueError("prefix-building action request requires prompt")
        if body.get("images") is None:
            raise ValueError("prefix-building action request requires images")
    elif body.get("prompt") is not None or body.get("images") is not None:
        raise ValueError("prefix reuse must not include prompt or images")

    model_state = body.get("state")
    raw_state = body.get("raw_state", model_state)
    normalized_state = (
        None if model_state is None else httpserver_manager.action_prepost_processor.normalize_state(model_state)
    )
    action_request = ActionRequest(
        request_id=body.get("request_id") or uuid.uuid4().hex,
        state=None if normalized_state is None else normalized_state.tolist(),
        raw_state=raw_state,
        noise=body.get("noise"),
        action_horizon=body.get("action_horizon"),
        action_dim=body.get("action_dim"),
        num_denoise_steps=body.get("num_denoise_steps"),
        timeout=float(body.get("timeout", 120.0)),
        metadata=body.get("metadata", {}),
        prefix_context=prefix_context,
    )
    request_config = action_request.validate(config)

    image_values = _ordered_images(body["images"], config.image_keys) if action_request.requires_prefix_inputs else []
    image_mask = body.get("image_mask")
    if image_mask is not None:
        if isinstance(image_mask, dict):
            mask_values = _ordered_images(image_mask, config.image_keys)
        else:
            mask_values = image_mask
        if not isinstance(mask_values, list):
            mask_values = [mask_values]
        if len(mask_values) != len(image_values):
            raise ValueError("image_mask length must match images")
        image_values = [image for image, enabled in zip(image_values, mask_values, strict=True) if bool(enabled)]
    if action_request.requires_prefix_inputs and not image_values:
        raise ValueError("action request requires at least one enabled image")

    output_plan = OutputPlan.from_outputs(body.get("outputs", ["action"]))
    if action_request.context_op is not PrefixContextOp.ONESHOT and output_plan.wants_text:
        raise ValueError("persistent prefix operations currently support action output only")
    multimodal_params = MultimodalParams(
        images=[_image_to_item(image) for image in image_values],
        action=action_request,
        outputs=output_plan.as_strings(),
        state=None if normalized_state is None else normalized_state.tolist(),
    )
    sampling_params = SamplingParams()
    suffix_length = request_config.action_horizon + (0 if request_config.is_pi05 else 1)
    if not output_plan.wants_text:
        requested_max_new_tokens = int(body.get("max_new_tokens", 1))
        if requested_max_new_tokens != 1:
            raise ValueError("action-only requests require max_new_tokens=1")
        max_new_tokens = 1
        ignore_eos = True
    else:
        max_new_tokens = int(body.get("max_new_tokens", suffix_length))
        ignore_eos = bool(body.get("ignore_eos", False))
    sampling_params.init(
        tokenizer=httpserver_manager.tokenizer,
        # Prefix-only action owners use one ordinary, internal completion
        # token so the unchanged LLM post/finish path can retire their ShmReq.
        # The action suffix has its own KV allocation and is not represented by
        # this text-token budget.
        max_new_tokens=max_new_tokens,
        n=1,
        best_of=1,
        ignore_eos=ignore_eos,
        # The VLM KV radix cache is disabled globally because this prefix is
        # bidirectional, but the ordinary visualserver embedding cache remains
        # valid and should still deduplicate identical camera frames.
        disable_prompt_cache=False,
    )
    sampling_params.verify()

    response = None
    text_parts = []
    finish_reason = None
    context_owner_identity = None
    context_owner_handed_to_response = False
    inference_prompt = (
        body["prompt"] if action_request.requires_prefix_inputs else [int(httpserver_manager.tokenizer.bos_token_id)]
    )
    try:
        async for _, text, metadata, finish_status in httpserver_manager.generate(
            inference_prompt,
            sampling_params,
            multimodal_params,
            request=request,
        ):
            if metadata.get("is_token", True):
                text_parts.append(text)
            if "action_response" in metadata:
                response = metadata["action_response"]
            private_owner = metadata.pop(
                "_action_context_owner_identity",
                None,
            )
            if private_owner is not None:
                if (
                    context_owner_identity is not None
                    and context_owner_identity != private_owner
                ):
                    raise RuntimeError(
                        "action response carried conflicting context owners"
                    )
                context_owner_identity = private_owner
            if finish_status.is_finished():
                finish_reason = finish_status.get_finish_reason()
        if output_plan.wants_action and response is None:
            raise RuntimeError("action request completed without a response")
        if action_request.persists_prefix and context_owner_identity is None:
            raise RuntimeError(
                "persistent context response is missing its owner handshake"
            )
        public_context = action_request.context_identity
        result_version = (
            None if response is None else response.get("prefix_context_version")
        )
        if action_request.persists_prefix and result_version is None:
            raise RuntimeError(
                "persistent context response is missing its committed version"
            )
        if (
            action_request.context_op is PrefixContextOp.CREATE
            and action_request.context_id is not None
            and result_version is not None
        ):
            public_context = PrefixContextIdentity(
                context_id=action_request.context_id,
                version=int(result_version),
                server_epoch=get_unique_server_name(),
            )
        if public_context is not None and result_version is not None:
            public_context = PrefixContextIdentity(
                context_id=public_context.context_id,
                version=int(result_version),
                server_epoch=public_context.server_epoch,
            )
        if public_context is not None and response is not None:
            response = dict(response)
            response.pop("prefix_context_version", None)
            response["prefix_context"] = public_context.to_dict()
        if response is not None and response.get("error_info"):
            if public_context is not None:
                # Prefix commit and action execution are separate outcomes.
                # Return the committed handle even when the first action fails
                # so callers can retry or close it instead of orphaning KV.
                final_content = response
                final_status_code = 500
            else:
                raise RuntimeError(response["error_info"])
        elif output_plan == OutputPlan.action_only():
            final_content = response
            final_status_code = 200
        else:
            result = {
                "request_id": action_request.request_id,
                "outputs": output_plan.as_strings(),
                "text": "".join(text_parts),
                "finish_reason": finish_reason,
            }
            if response is not None:
                result["action_response"] = response
            final_content = result
            final_status_code = 200

        if context_owner_identity is not None:
            final_response = _ContextOwnerJSONResponse(
                final_content,
                status_code=final_status_code,
                httpserver_manager=httpserver_manager,
                context_owner_identity=context_owner_identity,
            )
            context_owner_handed_to_response = True
        else:
            final_response = JSONResponse(
                final_content,
                status_code=final_status_code,
            )
        return final_response
    finally:
        if (
            context_owner_identity is not None
            and not context_owner_handed_to_response
        ):
            httpserver_manager.resolve_action_context_owner(
                context_owner_identity,
                delivered=False,
            )


async def close_prefix_context(
    request: Request,
    httpserver_manager,
    context_id: str,
) -> JSONResponse:
    params = request.query_params
    try:
        identity = PrefixContextIdentity(
            context_id=context_id,
            version=_parse_close_context_version(params["context_version"]),
            server_epoch=params["server_epoch"],
        )
    except KeyError as exc:
        raise ValueError("context close requires context_version and server_epoch") from exc
    timeout = float(params.get("timeout", 120.0))
    action_request = ActionRequest(
        state=None,
        request_id=params.get("request_id") or uuid.uuid4().hex,
        timeout=timeout,
        prefix_context=PrefixContextRef(PrefixContextOp.CLOSE, identity),
    )
    action_request.validate(httpserver_manager.vla_config)
    multimodal_params = MultimodalParams(
        action=action_request,
        outputs=["action"],
    )
    sampling_params = SamplingParams()
    sampling_params.init(
        tokenizer=httpserver_manager.tokenizer,
        max_new_tokens=1,
        n=1,
        best_of=1,
        ignore_eos=True,
        disable_prompt_cache=True,
    )
    sampling_params.verify()

    response = None
    async for _, _, metadata, _ in httpserver_manager.generate(
        [int(httpserver_manager.tokenizer.bos_token_id)],
        sampling_params,
        multimodal_params,
        request=request,
    ):
        if "action_response" in metadata:
            response = metadata["action_response"]
    if response is None:
        raise RuntimeError("context close completed without a response")
    if response.get("error_info"):
        raise RuntimeError(response["error_info"])
    return JSONResponse(
        {
            "prefix_context": identity.to_dict(),
            "status": "closed",
        }
    )
