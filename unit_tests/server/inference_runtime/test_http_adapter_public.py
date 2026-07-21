from types import SimpleNamespace

import pytest

from lightllm.server.inference_runtime.http_adapter import (
    resolve_request_output_plan,
)
from lightllm.server.inference_runtime.output_plan import OutputPlan


def _resolve(*, outputs=None, action=None, enabled=True, n=1, best_of=1):
    return resolve_request_output_plan(
        SimpleNamespace(outputs=outputs, action=action),
        SimpleNamespace(n=n, best_of=best_of),
        action_runtime_enabled=enabled,
    )


def test_legacy_request_without_action_defaults_to_text():
    assert _resolve(enabled=False) == OutputPlan.text_only()


def test_legacy_request_with_action_payload_defaults_to_action_only():
    assert _resolve(action=object()) == OutputPlan.action_only()


@pytest.mark.parametrize(
    ("outputs", "action", "enabled", "expected"),
    [
        (["text"], object(), False, OutputPlan.text_only()),
        (["action"], object(), True, OutputPlan.action_only()),
        (
            ["text", "action"],
            object(),
            True,
            OutputPlan.text_and_action(),
        ),
    ],
)
def test_explicit_output_plan_controls_requested_branches(outputs, action, enabled, expected):
    assert _resolve(outputs=outputs, action=action, enabled=enabled) == expected


@pytest.mark.parametrize(
    ("n", "best_of"),
    [
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_action_output_rejects_multiple_candidates(n, best_of):
    with pytest.raises(ValueError, match="n=1 and best_of=1"):
        _resolve(outputs=["action"], action=object(), n=n, best_of=best_of)


def test_action_output_requires_payload():
    with pytest.raises(ValueError, match="requires an action payload"):
        _resolve(outputs=["action"])


def test_action_output_requires_enabled_runtime():
    with pytest.raises(ValueError, match="not enabled"):
        _resolve(outputs=["action"], action=object(), enabled=False)
