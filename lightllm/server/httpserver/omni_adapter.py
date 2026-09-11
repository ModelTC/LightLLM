"""Translate public Omni requests to LightLLM and output-sidecar inputs.

This module mirrors :mod:`rl_controller`: HTTP concerns stay in the server
layer, while model packages expose only protocol and execution components.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import uuid
from typing import Any

import numpy as np
import orjson
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


_ACTION_OUTPUT = "action"
_TEXT_OUTPUT = "text"
_ALLOWED_OUTPUTS = frozenset({_ACTION_OUTPUT, _TEXT_OUTPUT})
_DEFAULT_TEXT_MAX_NEW_TOKENS = 64
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


class _ActionJSONResponse(JSONResponse):
    """Serialize the fixed action payload without stdlib JSON's float overhead."""

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


class _PersistentActionJSONResponse(_ActionJSONResponse):
    """Settle persistent-prefix ownership at the actual ASGI send boundary."""

    def __init__(self, content, *, httpserver_manager, group_request_id: int):
        super().__init__(content)
        self._httpserver_manager = httpserver_manager
        self._group_request_id = group_request_id

    async def _settle(self, delivered: bool) -> None:
        try:
            await self._httpserver_manager.settle_action_response_delivery(
                self._group_request_id,
                delivered=delivered,
            )
        except BaseException:
            # The response may already be on the wire.  Ownership cleanup must
            # be observable, but it must not replace the original send result.
            logger.exception(
                "failed to settle persistent action response group_request_id=%s",
                self._group_request_id,
            )

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            # Cancellation can immediately interrupt another await.  Shield a
            # separate cleanup task so it keeps running even if this request
            # task is already cancelled.
            cleanup = asyncio.create_task(self._settle(False))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass
            raise
        else:
            await self._settle(True)


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


def _resolve_prefix_context(body: dict[str, Any]) -> PrefixContextRef:
    from lightllm.server.generationserver.action.protocol import (
        PrefixContextIdentity,
        PrefixContextOp,
        PrefixContextRef,
    )

    if not isinstance(body, dict):
        raise ValueError("action request body must be an object")

    if "prefix_context" in body:
        conflicting = _FLAT_CONTEXT_FIELDS.intersection(body)
        if conflicting:
            raise ValueError(
                "prefix_context cannot be combined with flat context fields: "
                + ", ".join(sorted(conflicting))
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
            context_id=(
                body["context_id"] if "context_id" in body else uuid.uuid4().hex
            ),
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

    return PrefixContextRef(
        op,
        PrefixContextIdentity(
            context_id=body["context_id"],
            version=_json_context_version(body["context_version"]),
            server_epoch=body["server_epoch"],
        ),
    )


def _parse_outputs(value: Any) -> tuple[list[str], set[str]]:
    if value is None:
        outputs = [_ACTION_OUTPUT]
    elif isinstance(value, str):
        outputs = [value]
    elif isinstance(value, (list, tuple)):
        outputs = list(value)
    else:
        raise ValueError("outputs must be a string or a list")

    if not outputs:
        raise ValueError("outputs must not be empty")
    if any(not isinstance(output, str) for output in outputs):
        raise ValueError("outputs entries must be strings")
    if len(outputs) != len(set(outputs)):
        raise ValueError("outputs must not contain duplicates")
    unsupported = set(outputs).difference(_ALLOWED_OUTPUTS)
    if unsupported:
        raise ValueError("unsupported outputs: " + ", ".join(sorted(unsupported)))
    if _ACTION_OUTPUT not in outputs:
        raise ValueError("the VLA action endpoint requires action output")
    return outputs, set(outputs)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _request_id(value: Any) -> str:
    if value is None:
        return uuid.uuid4().hex
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_id must be a non-empty string")
    return value


def _image_to_item(value: Any, *, input_key: str | None = None) -> dict[str, Any]:
    """Convert the public wire representation without model preprocessing."""

    def item(image_type: str, data: Any) -> dict[str, Any]:
        result = {"type": image_type, "data": data}
        if input_key is not None:
            result["input_key"] = input_key
        return result

    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return item("url", value)
        if value.startswith("data:"):
            _, separator, encoded = value.partition(",")
            if not separator:
                raise ValueError("image data URL is missing its base64 payload")
        else:
            encoded = value
        base64.b64decode(encoded, validate=True)
        return item("base64", encoded)

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
    # This is transport encoding only. Use lossless PNG so the visual model
    # receives the same uint8 pixels the caller supplied.
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return item("base64", base64.b64encode(buffer.getvalue()).decode("ascii"))


def _is_single_image_array(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.dtype != object and array.ndim == 3 and array.shape[-1] in {3, 4}


def _select_image_inputs(images: Any, image_mask: Any) -> list[tuple[str | None, Any]]:
    """Apply the wire mask while preserving optional semantic input names."""

    if isinstance(images, dict):
        entries = list(images.items())
        if image_mask is None:
            return entries
        if isinstance(image_mask, dict):
            if set(image_mask) != set(images):
                raise ValueError("image_mask keys must exactly match images keys")
            return [(key, image) for key, image in entries if bool(image_mask[key])]
        mask_values = image_mask if isinstance(image_mask, list) else [image_mask]
        if len(mask_values) != len(entries):
            raise ValueError("image_mask length must match images")
        return [
            (key, image)
            for (key, image), enabled in zip(entries, mask_values, strict=True)
            if bool(enabled)
        ]

    image_values = (
        [images]
        if not isinstance(images, list) or _is_single_image_array(images)
        else images
    )
    if image_mask is None:
        return [(None, image) for image in image_values]
    if isinstance(image_mask, dict):
        raise ValueError("dictionary image_mask requires dictionary images")
    mask_values = image_mask if isinstance(image_mask, list) else [image_mask]
    if len(mask_values) != len(image_values):
        raise ValueError("image_mask length must match images")
    return [
        (None, image)
        for image, enabled in zip(image_values, mask_values, strict=True)
        if bool(enabled)
    ]


def _make_action_request(
    body: dict[str, Any], prefix_context: PrefixContextRef
) -> ActionRequest:
    from lightllm.server.generationserver.action.protocol import ActionRequest

    state = body.get("state")
    return ActionRequest(
        request_id=_request_id(body.get("request_id")),
        state=state,
        domain=body.get("domain"),
        raw_state=body.get("raw_state", state),
        noise=body.get("noise"),
        action_horizon=body.get("action_horizon"),
        action_dim=body.get("action_dim"),
        num_denoise_steps=body.get("num_denoise_steps"),
        seed=body.get("seed"),
        timeout=_positive_float(body.get("timeout", 120.0), "timeout"),
        prefix_context=prefix_context,
    )


def _sampling_params(
    body: dict[str, Any], outputs: set[str], tokenizer
) -> SamplingParams:
    wants_text = _TEXT_OUTPUT in outputs
    default_max_tokens = _DEFAULT_TEXT_MAX_NEW_TOKENS if wants_text else 1
    max_new_tokens = _positive_int(
        body.get("max_new_tokens", default_max_tokens), "max_new_tokens"
    )
    if not wants_text and max_new_tokens != 1:
        raise ValueError("action-only requests require max_new_tokens=1")
    ignore_eos = body.get("ignore_eos", not wants_text)
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be a boolean")

    params = SamplingParams()
    params.init(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        n=1,
        best_of=1,
        ignore_eos=ignore_eos,
        disable_prompt_cache=False,
    )
    params.verify()
    return params


def _action_result_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, dict):
        raise RuntimeError("action broker returned an invalid response")
    return result


def _check_action_result(result: dict[str, Any]) -> None:
    if result.get("error_info"):
        raise RuntimeError(str(result["error_info"]))


async def generate_actions(request: Request, httpserver_manager) -> JSONResponse:
    from lightllm.server.generationserver.action.protocol import PrefixContextOp
    from lightllm.server.generationserver.action.io import (
        adapt_action_domain_body,
        adapt_action_domain_images,
    )

    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("action request body must be an object")

    runtime_config = getattr(
        getattr(httpserver_manager, "args", None),
        "generation_runtime_config",
        None,
    )
    body = adapt_action_domain_body(body, runtime_config)

    outputs, output_set = _parse_outputs(body.get("outputs"))
    prefix_context = _resolve_prefix_context(body)
    if prefix_context.op is PrefixContextOp.CLOSE:
        raise ValueError("use DELETE /v1/vla/contexts/{context_id} to close a context")
    if prefix_context.op is PrefixContextOp.REUSE and output_set != {_ACTION_OUTPUT}:
        raise ValueError("prefix reuse supports action output only")

    action_request = _make_action_request(body, prefix_context)
    if not action_request.requires_prefix_inputs:
        if body.get("prompt") is not None or body.get("images") is not None:
            raise ValueError("prefix reuse must not include prompt or images")
        result = _action_result_dict(
            await httpserver_manager.submit_action_context(action_request)
        )
        _check_action_result(result)
        return _ActionJSONResponse(result)

    if body.get("prompt") is None:
        raise ValueError("prefix-building action request requires prompt")
    if body.get("images") is None:
        raise ValueError("prefix-building action request requires images")
    image_inputs = _select_image_inputs(body["images"], body.get("image_mask"))
    image_inputs = adapt_action_domain_images(
        action_request.domain,
        image_inputs,
        runtime_config,
    )
    if not image_inputs:
        raise ValueError("action request requires at least one enabled image")

    multimodal_params = MultimodalParams(
        images=[
            _image_to_item(image, input_key=input_key)
            for input_key, image in image_inputs
        ],
        action=action_request,
        outputs=outputs,
        # State remains raw here. Pi0/Pi0.5 adapters own normalization and
        # tokenization because those operations are model-specific.
        state=action_request.state,
    )
    sampling_params = _sampling_params(body, output_set, httpserver_manager.tokenizer)

    action_response = None
    text_parts: list[str] = []
    finish_reason = None
    group_request_id = None
    async for sub_request_id, text, metadata, finish_status in httpserver_manager.generate(
        body["prompt"],
        sampling_params,
        multimodal_params,
        request=request,
    ):
        if group_request_id is None:
            group_request_id = sub_request_id
        elif group_request_id != sub_request_id:
            raise RuntimeError("action response mixed multiple request groups")
        if metadata.get("is_token", True):
            text_parts.append(text)
        if "action_response" in metadata:
            action_response = _action_result_dict(metadata["action_response"])
        if finish_status.is_finished():
            finish_reason = finish_status.get_finish_reason()

    try:
        if action_response is None:
            raise RuntimeError("action request completed without a response")
        _check_action_result(action_response)
        if action_request.persists_prefix and "prefix_context" not in action_response:
            raise RuntimeError(
                "persistent action response is missing its prefix context"
            )

        def build_response(content):
            if not action_request.persists_prefix:
                return _ActionJSONResponse(content)
            if group_request_id is None:
                raise RuntimeError(
                    "persistent action response is missing its request group"
                )
            return _PersistentActionJSONResponse(
                content,
                httpserver_manager=httpserver_manager,
                group_request_id=group_request_id,
            )

        if output_set == {_ACTION_OUTPUT}:
            return build_response(action_response)
        return build_response(
            {
                "request_id": action_request.request_id,
                "outputs": outputs,
                "text": "".join(text_parts),
                "finish_reason": finish_reason,
                "action_response": action_response,
            }
        )
    except BaseException:
        # A persistent context may already have committed even though response
        # validation/serialization failed before an ASGI Response was built.
        if action_request.persists_prefix and group_request_id is not None:
            await httpserver_manager.settle_action_response_delivery(
                group_request_id,
                delivered=False,
            )
        raise


async def close_prefix_context(
    request: Request,
    httpserver_manager,
    context_id: str,
) -> JSONResponse:
    from lightllm.server.generationserver.action.protocol import (
        ActionRequest,
        PrefixContextIdentity,
        PrefixContextOp,
        PrefixContextRef,
    )

    params = request.query_params
    try:
        identity = PrefixContextIdentity(
            context_id=context_id,
            version=_parse_close_context_version(params["context_version"]),
            server_epoch=params["server_epoch"],
        )
    except KeyError as exc:
        raise ValueError(
            "context close requires context_version and server_epoch"
        ) from exc

    action_request = ActionRequest(
        state=None,
        request_id=_request_id(params.get("request_id")),
        timeout=_positive_float(params.get("timeout", 120.0), "timeout"),
        prefix_context=PrefixContextRef(PrefixContextOp.CLOSE, identity),
    )
    result = _action_result_dict(
        await httpserver_manager.submit_action_context(action_request)
    )
    _check_action_result(result)
    return _ActionJSONResponse(
        {
            "prefix_context": identity.to_dict(),
            "status": "closed",
        }
    )


def _image_generation_input(body: dict[str, Any], prefix_roles: tuple[str, ...]):
    from lightllm.server.generationserver.image.protocol import ImageGenerationInput

    return ImageGenerationInput(
        prefix_roles=prefix_roles,
        generation_id=uuid.uuid4().hex,
        request_id=body.get("request_id") or uuid.uuid4().hex,
        width=body.get("width", 1024),
        height=body.get("height", 1024),
        steps=body.get("steps", 50),
        guidance_scale=body.get("guidance_scale", 4.0),
        seed=body.get("seed", 42),
        num_outputs=body.get("num_outputs", 1),
        cfg_interval=tuple(body.get("cfg_interval", (-1.0, 2.0))),
        cfg_norm=body.get("cfg_norm", "global"),
        timestep_shift=body.get("timestep_shift", 3.0),
        timeout=body.get("timeout", 300.0),
    )


def _normalize_image(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if set(value) != {"type", "data"} or value["type"] not in {"url", "base64"}:
            raise ValueError("image objects must contain type=url|base64 and data")
        value = dict(value)
        if value["type"] == "base64":
            base64.b64decode(value["data"], validate=True)
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("images must be URL/base64 strings or {type, data} objects")
    if value.startswith(("http://", "https://")):
        return {"type": "url", "data": value}
    encoded = value.partition(",")[2] if value.startswith("data:") else value
    base64.b64decode(encoded, validate=True)
    return {"type": "base64", "data": encoded}


def _image_sampling_params(tokenizer) -> SamplingParams:
    params = SamplingParams()
    params.init(
        tokenizer,
        do_sample=False,
        max_new_tokens=1,
        n=1,
        best_of=1,
        ignore_eos=True,
        disable_prompt_cache=True,
        add_special_tokens=True,
    )
    params.verify()
    return params


async def generate_images(request: Request, httpserver_manager) -> JSONResponse:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("generation request body must be an object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generation prompt must be a non-empty string")
    raw_images = body.get("images", [])
    if not isinstance(raw_images, list):
        raw_images = [raw_images]
    images = [_normalize_image(image) for image in raw_images]
    generation = _image_generation_input(
        body,
        httpserver_manager.tokenizer.generation_prefix_roles,
    )
    prefix_inputs = httpserver_manager.tokenizer.build_generation_prefixes(
        prompt=prompt,
        images=images,
        generation=generation,
    )

    async def run_prefix(prefix_input):
        response = None
        multimodal = MultimodalParams(
            images=[dict(image) for image in prefix_input.images],
            generation=generation,
            generation_role=prefix_input.role,
            outputs=["image"],
        )
        async for _, _, metadata, _ in httpserver_manager.generate(
            prefix_input.prompt,
            _image_sampling_params(httpserver_manager.tokenizer),
            multimodal,
            request=request,
        ):
            if "generation_response" in metadata:
                response = metadata["generation_response"]
        if response is None:
            raise RuntimeError(
                f"generation prefix {prefix_input.role} completed without a result"
            )
        return response

    tasks = [asyncio.create_task(run_prefix(prefix)) for prefix in prefix_inputs]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    result = results[0]
    if any(other.generation_id != result.generation_id for other in results[1:]):
        raise RuntimeError("generation prefix results do not match")
    if not result.ok:
        raise RuntimeError(result.error_info or "generation backend failed")
    return JSONResponse(result.to_dict())
