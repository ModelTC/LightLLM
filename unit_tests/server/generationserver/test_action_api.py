import asyncio
import json
from types import SimpleNamespace

import pytest

from lightllm.server.httpserver.omni_adapter import (
    _parse_close_context_version,
    _parse_outputs,
    _resolve_prefix_context,
    _select_image_inputs,
    close_prefix_context,
    generate_actions,
)
from lightllm.server.generationserver.action.protocol import PrefixContextOp
from lightllm.server.core.objs.req import FinishStatus


class _Request:
    def __init__(self, body=None, query_params=None):
        self.body = body
        self.query_params = query_params or {}

    async def json(self):
        return self.body


class _Manager:
    tokenizer = None

    def __init__(self, events=(), context_result=None, runtime_config=None):
        self.events = events
        self.context_result = context_result
        if runtime_config is not None:
            self.args = SimpleNamespace(generation_runtime_config=runtime_config)
        self.generate_call = None
        self.context_requests = []
        self.delivery_settlements = []

    async def generate(self, prompt, sampling_params, multimodal_params, *, request):
        self.generate_call = SimpleNamespace(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=request,
        )
        for event in self.events:
            yield event

    async def submit_action_context(self, action_request):
        self.context_requests.append(action_request)
        return self.context_result

    async def settle_action_response_delivery(self, group_request_id, *, delivered):
        self.delivery_settlements.append((group_request_id, delivered))
        return True


def _body(outputs=...):
    body = {
        "request_id": "request-17",
        "prompt": "move the block",
        "state": [0.25, -0.5],
        "images": {"camera": [[[0, 0, 0]]]},
    }
    if outputs is not ...:
        body["outputs"] = outputs
    return body


def _finish(status=FinishStatus.FINISHED_STOP):
    return FinishStatus(status)


def _response(response):
    return json.loads(response.body)


def test_persistent_context_wire_modes_are_explicit():
    created = _resolve_prefix_context({"persist_prefix": True, "context_id": "robot-7"})
    assert created.op is PrefixContextOp.CREATE
    assert created.identity is None
    assert created.context_id == "robot-7"

    generated = _resolve_prefix_context({"persist_prefix": True})
    assert generated.op is PrefixContextOp.CREATE
    assert generated.context_id

    reused = _resolve_prefix_context(
        {
            "context_id": "robot-7",
            "context_version": 1,
            "server_epoch": "test-epoch",
        }
    )
    assert reused.op is PrefixContextOp.REUSE
    assert reused.identity.to_dict() == {
        "context_id": "robot-7",
        "version": 1,
        "server_epoch": "test-epoch",
    }

    replaced = _resolve_prefix_context(
        {
            "context_id": "robot-7",
            "context_version": 1,
            "server_epoch": "test-epoch",
            "replace_prefix": True,
        }
    )
    assert replaced.op is PrefixContextOp.REPLACE
    assert replaced.identity == reused.identity


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"persist_prefix": 1}, "persist_prefix must be a boolean"),
        ({"replace_prefix": None}, "replace_prefix must be a boolean"),
        (
            {"persist_prefix": True, "replace_prefix": True},
            "cannot both be true",
        ),
        (
            {"persist_prefix": True, "context_version": 1},
            "create must not include existing-identity fields",
        ),
        ({"replace_prefix": True}, "replace requires exact identity fields"),
        ({"context_id": "robot-7"}, "reuse requires exact identity fields"),
        (
            {
                "prefix_context": {"op": "create", "context_id": "robot-7"},
                "persist_prefix": False,
            },
            "cannot be combined with flat context fields",
        ),
    ],
)
def test_context_wire_rejects_malformed_or_mixed_modes(body, message):
    with pytest.raises(ValueError, match=message):
        _resolve_prefix_context(body)


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 0, -1])
def test_context_identity_requires_a_positive_json_integer(version):
    with pytest.raises(ValueError, match="positive JSON integer"):
        _resolve_prefix_context(
            {
                "context_id": "robot-7",
                "context_version": version,
                "server_epoch": "test-epoch",
            }
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("001", 1), ("42", 42)],
)
def test_close_context_version_accepts_positive_decimal_strings(value, expected):
    assert _parse_close_context_version(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "0", "-1", "+1", " 1", "1 ", "1.0", "1e0", "True", 1, True, None],
)
def test_close_context_version_rejects_other_values(value):
    with pytest.raises(ValueError, match="positive decimal integer"):
        _parse_close_context_version(value)


def test_outputs_are_a_small_explicit_protocol():
    assert _parse_outputs(None) == (["action"], {"action"})
    assert _parse_outputs(["text", "action"]) == (
        ["text", "action"],
        {"text", "action"},
    )
    with pytest.raises(ValueError, match="requires action"):
        _parse_outputs(["text"])
    with pytest.raises(ValueError, match="duplicates"):
        _parse_outputs(["action", "action"])
    with pytest.raises(ValueError, match="unsupported"):
        _parse_outputs(["action", "image"])


def test_image_dictionary_order_and_mask_are_preserved():
    images = {"wrist": "wrist-image", "base": "base-image"}
    assert _select_image_inputs(images, None) == [
        ("wrist", "wrist-image"),
        ("base", "base-image"),
    ]
    with pytest.raises(ValueError, match="keys must exactly match"):
        _select_image_inputs(images, {"base": True})
    assert _select_image_inputs(images, {"wrist": False, "base": True}) == [
        ("base", "base-image")
    ]


def test_x2v_libero_domain_maps_cameras_and_public_action_shape():
    runtime_config = {
        "image_keys": (
            "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        ),
        "domain_aliases": {"x2v_libero": "libero"},
        "domain_profiles": {
            "libero": {
                "state_dim": 8,
                "action_dim": 7,
                "image_map": {
                    "agentview": "observation.images.base_0_rgb",
                    "wrist": "observation.images.left_wrist_0_rgb",
                },
                "ignored_images": ["frontview"],
                "required_images": ["observation.images.base_0_rgb"],
            }
        },
    }
    action_response = {
        "request_id": "request-17",
        "actions": [[0.0] * 7],
        "error_info": None,
    }
    manager = _Manager(
        [
            (
                17,
                "",
                {"is_token": False, "action_response": action_response},
                _finish(),
            )
        ],
        runtime_config=runtime_config,
    )
    image = [[[0, 0, 0]]]
    body = {
        "request_id": "request-17",
        "domain_name": "x2v_libero",
        "prompt": "pick up the block",
        "state": [0.0] * 8,
        "images": {
            "wrist": image,
            "frontview": image,
            "agentview": image,
        },
    }

    response = asyncio.run(generate_actions(_Request(body), manager))

    assert _response(response) == action_response
    call = manager.generate_call
    assert call.multimodal_params.action.domain == "libero"
    assert call.multimodal_params.action.action_dim == 7
    assert [item.input_key for item in call.multimodal_params.images] == [
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
    ]


def test_domain_request_rejects_ambiguous_camera_mapping():
    runtime_config = {
        "image_keys": ("observation.images.base_0_rgb",),
        "domain_profiles": {
            "libero": {
                "state_dim": 8,
                "action_dim": 7,
                "image_map": {
                    "agentview": "observation.images.base_0_rgb",
                    "frontview": "observation.images.base_0_rgb",
                },
                "required_images": ["observation.images.base_0_rgb"],
            }
        },
    }
    manager = _Manager(runtime_config=runtime_config)
    image = [[[0, 0, 0]]]
    body = {
        "domain": "libero",
        "prompt": "move",
        "state": [0.0] * 8,
        "images": {"agentview": image, "frontview": image},
    }

    with pytest.raises(ValueError, match="multiple domain cameras"):
        asyncio.run(generate_actions(_Request(body), manager))


def test_cold_action_request_keeps_raw_inputs_and_returns_bare_action():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0], [3.0, 4.0]],
        "error_info": None,
    }
    manager = _Manager(
        [
            (
                17,
                "ignored",
                {"is_token": False, "action_response": action_response},
                _finish(),
            )
        ]
    )

    response = asyncio.run(generate_actions(_Request(_body()), manager))

    assert _response(response) == action_response
    call = manager.generate_call
    assert call.prompt == "move the block"
    assert call.multimodal_params.outputs == ["action"]
    assert call.multimodal_params.state == [0.25, -0.5]
    assert call.multimodal_params.action.state == [0.25, -0.5]
    assert call.multimodal_params.action.raw_state == [0.25, -0.5]
    assert call.multimodal_params.images[0].input_key == "camera"
    assert call.sampling_params.max_new_tokens == 1
    assert call.sampling_params.ignore_eos


def test_cold_text_and_action_response_is_joined_directly():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0]],
        "error_info": None,
    }
    manager = _Manager(
        [
            (17, "moving", {"is_token": True}, _finish(FinishStatus.NO_FINISH)),
            (
                17,
                "not text",
                {"is_token": False, "action_response": action_response},
                _finish(),
            ),
        ]
    )

    response = asyncio.run(
        generate_actions(_Request(_body(["text", "action"])), manager)
    )

    assert _response(response) == {
        "request_id": "request-17",
        "outputs": ["text", "action"],
        "text": "moving",
        "finish_reason": "stop",
        "action_response": action_response,
    }
    assert manager.generate_call.sampling_params.max_new_tokens == 64
    assert not manager.generate_call.sampling_params.ignore_eos


def test_persistent_cold_request_uses_context_returned_by_action_manager():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0]],
        "error_info": None,
        "prefix_context": {
            "context_id": "robot-7",
            "version": 1,
            "server_epoch": "test-epoch",
        },
    }
    manager = _Manager(
        [(17, "", {"is_token": False, "action_response": action_response}, _finish())]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-7"})

    response = asyncio.run(generate_actions(_Request(body), manager))

    assert _response(response) == action_response
    assert not hasattr(manager, "action_prepost_processor")
    assert not hasattr(manager, "vla_config")


def test_persistent_context_is_settled_only_after_response_send():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0]],
        "error_info": None,
        "prefix_context": {
            "context_id": "robot-7",
            "version": 1,
            "server_epoch": "test-epoch",
        },
    }
    manager = _Manager(
        [(17, "", {"is_token": False, "action_response": action_response}, _finish())]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-7"})
    response = asyncio.run(generate_actions(_Request(body), manager))

    assert manager.delivery_settlements == []

    async def send(_message):
        return None

    asyncio.run(response({"type": "http"}, None, send))
    assert manager.delivery_settlements == [(17, True)]


def test_persistent_context_is_discarded_when_response_send_fails():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0]],
        "error_info": None,
        "prefix_context": {
            "context_id": "robot-7",
            "version": 1,
            "server_epoch": "test-epoch",
        },
    }
    manager = _Manager(
        [(17, "", {"is_token": False, "action_response": action_response}, _finish())]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-7"})
    response = asyncio.run(generate_actions(_Request(body), manager))

    async def send(_message):
        raise RuntimeError("client disconnected during response send")

    with pytest.raises(RuntimeError, match="client disconnected"):
        asyncio.run(response({"type": "http"}, None, send))
    assert manager.delivery_settlements == [(17, False)]


def test_persistent_context_is_discarded_when_response_validation_fails():
    manager = _Manager(
        [
            (
                17,
                "",
                {
                    "is_token": False,
                    "action_response": {
                        "request_id": "request-17",
                        "actions": [[1.0, 2.0]],
                        "error_info": None,
                    },
                },
                _finish(),
            )
        ]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-7"})

    with pytest.raises(RuntimeError, match="missing its prefix context"):
        asyncio.run(generate_actions(_Request(body), manager))
    assert manager.delivery_settlements == [(17, False)]


def test_hot_context_request_bypasses_generate_and_router_inputs():
    context = {
        "context_id": "robot-7",
        "version": 1,
        "server_epoch": "test-epoch",
    }
    result = {
        "request_id": "tick-2",
        "actions": [[1.0, 2.0]],
        "error_info": None,
        "prefix_context": context,
    }
    manager = _Manager(context_result=result)
    body = {
        "request_id": "tick-2",
        "state": [0.5, -0.25],
        "prefix_context": {"op": "reuse", "identity": context},
    }

    response = asyncio.run(generate_actions(_Request(body), manager))

    assert _response(response) == result
    assert manager.generate_call is None
    assert len(manager.context_requests) == 1
    assert manager.context_requests[0].state == [0.5, -0.25]


def test_close_context_uses_direct_action_broker():
    manager = _Manager(context_result={"error_info": None})
    request = _Request(
        query_params={
            "context_version": "3",
            "server_epoch": "test-epoch",
            "request_id": "close-1",
        }
    )

    response = asyncio.run(close_prefix_context(request, manager, "robot-7"))

    assert _response(response) == {
        "prefix_context": {
            "context_id": "robot-7",
            "version": 3,
            "server_epoch": "test-epoch",
        },
        "status": "closed",
    }
    assert manager.generate_call is None
    assert manager.context_requests[0].context_op is PrefixContextOp.CLOSE


def test_action_endpoint_rejects_non_object_json_body():
    with pytest.raises(ValueError, match="body must be an object"):
        asyncio.run(generate_actions(_Request([]), _Manager()))
