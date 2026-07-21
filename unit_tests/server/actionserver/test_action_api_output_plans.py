import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lightllm.server.actionserver.api import (
    _parse_close_context_version,
    _resolve_prefix_context,
    generate_actions,
)
from lightllm.server.actionserver.objs import ActionTaskIdentity, PrefixContextOp
from lightllm.server.core.objs.req import FinishStatus


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class _Config:
    model_type = SimpleNamespace(value="pi05")
    image_keys = ("camera",)
    max_state_dim = 8
    action_dim = 2
    action_horizon = 2
    num_denoise_steps = 1
    is_pi05 = True

    def with_overrides(self, *, action_dim=None, action_horizon=None, num_denoise_steps=None):
        return SimpleNamespace(
            action_dim=self.action_dim if action_dim is None else action_dim,
            action_horizon=(self.action_horizon if action_horizon is None else action_horizon),
            num_denoise_steps=(self.num_denoise_steps if num_denoise_steps is None else num_denoise_steps),
            is_pi05=True,
        )


class _Manager:
    vla_config = _Config()
    tokenizer = None

    def __init__(self, events):
        self.events = events
        self.action_prepost_processor = SimpleNamespace(
            normalize_state=lambda state: np.asarray(state, dtype=np.float32)
        )
        self.generate_call = None
        self.owner_resolutions = []

    async def generate(self, prompt, sampling_params, multimodal_params, *, request):
        self.generate_call = SimpleNamespace(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=request,
        )
        for event in self.events:
            yield event

    def resolve_action_context_owner(self, identity, *, delivered):
        self.owner_resolutions.append((identity, delivered))
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


async def _send_asgi_response(response, *, fail_body=False):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)
        if fail_body and message["type"] == "http.response.body":
            raise ConnectionError("client disconnected during response body")

    await response(
        {"type": "http", "method": "POST", "path": "/v1/vla/actions"},
        receive,
        send,
    )
    return messages


def test_persistent_context_wire_modes_are_explicit(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test-epoch")
    from lightllm.utils.envs_utils import get_unique_server_name

    get_unique_server_name.cache_clear()
    created = _resolve_prefix_context({"persist_prefix": True, "context_id": "robot-7"})
    assert created.op is PrefixContextOp.CREATE
    assert created.identity is None
    assert created.context_id == "robot-7"
    assert created.to_dict() == {
        "op": "create",
        "context_id": "robot-7",
    }
    generated = _resolve_prefix_context({"persist_prefix": True})
    assert generated.op is PrefixContextOp.CREATE
    assert generated.identity is None
    assert isinstance(generated.context_id, str)
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
    assert reused.context_id is None

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
    assert replaced.context_id is None
    get_unique_server_name.cache_clear()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"persist_prefix": 1}, "persist_prefix must be a boolean"),
        ({"persist_prefix": "true"}, "persist_prefix must be a boolean"),
        ({"replace_prefix": 1}, "replace_prefix must be a boolean"),
        ({"replace_prefix": None}, "replace_prefix must be a boolean"),
        (
            {"persist_prefix": True, "replace_prefix": True},
            "cannot both be true",
        ),
        (
            {"persist_prefix": True, "context_version": 1},
            "create must not include existing-identity fields",
        ),
        (
            {"persist_prefix": True, "server_epoch": "epoch"},
            "create must not include existing-identity fields",
        ),
        (
            {"replace_prefix": True},
            "replace requires exact identity fields",
        ),
        (
            {"context_id": "robot-7"},
            "reuse requires exact identity fields",
        ),
        (
            {"context_version": 1, "server_epoch": "epoch"},
            "reuse requires exact identity fields",
        ),
        (
            {
                "prefix_context": {"op": "create", "context_id": "robot-7"},
                "persist_prefix": False,
            },
            "cannot be combined with flat context fields",
        ),
        (
            {"persist_prefix": True, "context_id": None},
            "requires a context_id",
        ),
        (
            {"persist_prefix": True, "context_id": ""},
            "context_id must be a non-empty string",
        ),
        (
            {"persist_prefix": True, "context_id": 0},
            "context_id must be a non-empty string",
        ),
    ],
)
def test_flat_context_wire_rejects_malformed_or_mixed_modes(body, message):
    with pytest.raises(ValueError, match=message):
        _resolve_prefix_context(body)


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 0, -1])
def test_flat_context_identity_requires_a_positive_json_integer_version(version):
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
def test_close_context_version_accepts_positive_decimal_query_strings(value, expected):
    assert _parse_close_context_version(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "0", "-1", "+1", " 1", "1 ", "1.0", "1e0", "True", 1, True, None],
)
def test_close_context_version_rejects_non_decimal_or_truncated_values(value):
    with pytest.raises(ValueError, match="positive decimal integer"):
        _parse_close_context_version(value)


def test_action_endpoint_legacy_default_returns_bare_action_response():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0], [3.0, 4.0]],
        "error_info": None,
    }
    manager = _Manager([(17, "ignored", {"is_token": False, "action_response": action_response}, _finish())])
    request = _Request(_body())

    response = asyncio.run(generate_actions(request, manager))

    assert _response(response) == action_response
    assert manager.generate_call.multimodal_params.outputs == ["action"]
    assert manager.generate_call.sampling_params.ignore_eos


def test_action_endpoint_explicit_text_only_assembles_text_response():
    manager = _Manager(
        [
            (17, "move", {"is_token": True}, _finish(FinishStatus.NO_FINISH)),
            (17, " complete", {"is_token": True}, _finish()),
        ]
    )
    request = _Request(_body(["text"]))

    response = asyncio.run(generate_actions(request, manager))

    assert _response(response) == {
        "request_id": "request-17",
        "outputs": ["text"],
        "text": "move complete",
        "finish_reason": "stop",
    }
    assert manager.generate_call.multimodal_params.outputs == ["text"]
    assert not manager.generate_call.sampling_params.ignore_eos


def test_action_endpoint_text_and_action_assembles_both_outputs():
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0], [3.0, 4.0]],
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
    request = _Request(_body(["text", "action"]))

    response = asyncio.run(generate_actions(request, manager))

    assert _response(response) == {
        "request_id": "request-17",
        "outputs": ["text", "action"],
        "text": "moving",
        "finish_reason": "stop",
        "action_response": action_response,
    }
    assert manager.generate_call.multimodal_params.outputs == ["text", "action"]


def test_persistent_action_delivers_owner_only_after_public_response_assembly(
    monkeypatch,
):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test-epoch")
    from lightllm.utils.envs_utils import get_unique_server_name

    get_unique_server_name.cache_clear()
    owner = ActionTaskIdentity(3, 17, 17)
    action_response = {
        "request_id": "request-17",
        "actions": [[1.0, 2.0]],
        "error_info": None,
        "prefix_context_version": 2,
    }
    manager = _Manager(
        [
            (
                17,
                "",
                {
                    "is_token": False,
                    "action_response": action_response,
                    "_action_context_owner_identity": owner,
                },
                _finish(),
            )
        ]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-7"})

    response = asyncio.run(generate_actions(_Request(body), manager))

    assert _response(response)["prefix_context"] == {
        "context_id": "robot-7",
        "version": 2,
        "server_epoch": "test-epoch",
    }
    assert manager.owner_resolutions == []
    asyncio.run(_send_asgi_response(response))
    assert manager.owner_resolutions == [(owner, True)]
    get_unique_server_name.cache_clear()


def test_persistent_action_discards_owner_when_response_assembly_fails():
    owner = ActionTaskIdentity(4, 18, 18)
    manager = _Manager(
        [
            (
                18,
                "",
                {
                    "is_token": False,
                    "action_response": {
                        "request_id": "request-18",
                        "actions": [[1.0, 2.0]],
                        "error_info": None,
                    },
                    "_action_context_owner_identity": owner,
                },
                _finish(),
            )
        ]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-8"})

    with pytest.raises(RuntimeError, match="missing its committed version"):
        asyncio.run(generate_actions(_Request(body), manager))

    assert manager.owner_resolutions == [(owner, False)]


def test_action_endpoint_rejects_non_object_json_body():
    manager = _Manager([])
    with pytest.raises(ValueError, match="body must be an object"):
        asyncio.run(generate_actions(_Request([]), manager))


def test_persistent_action_discards_owner_when_asgi_body_send_fails(
    monkeypatch,
):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test-epoch")
    from lightllm.utils.envs_utils import get_unique_server_name

    get_unique_server_name.cache_clear()
    owner = ActionTaskIdentity(5, 19, 19)
    manager = _Manager(
        [
            (
                19,
                "",
                {
                    "is_token": False,
                    "action_response": {
                        "request_id": "request-19",
                        "actions": [[1.0, 2.0]],
                        "error_info": None,
                        "prefix_context_version": 3,
                    },
                    "_action_context_owner_identity": owner,
                },
                _finish(),
            )
        ]
    )
    body = _body()
    body.update({"persist_prefix": True, "context_id": "robot-9"})
    response = asyncio.run(generate_actions(_Request(body), manager))

    with pytest.raises(ConnectionError, match="disconnected"):
        asyncio.run(_send_asgi_response(response, fail_body=True))

    assert manager.owner_resolutions == [(owner, False)]
    get_unique_server_name.cache_clear()
