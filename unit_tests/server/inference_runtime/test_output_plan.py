import pytest

from lightllm.server.inference_runtime import OutputKind, OutputPlan
from lightllm.server.multimodal_params import MultimodalParams


def test_omitted_output_plan_preserves_legacy_text_default():
    plan = OutputPlan.from_outputs()

    assert plan == OutputPlan.text_only()
    assert plan.wants_text
    assert not plan.wants_action
    assert plan.as_strings() == ("text",)


def test_legacy_action_payload_resolves_to_action_only():
    plan = OutputPlan.resolve(legacy_action_requested=True)

    assert plan == OutputPlan.action_only()
    assert not plan.wants_text
    assert plan.wants_action


def test_explicit_outputs_override_legacy_request_shape():
    plan = OutputPlan.resolve(["text", OutputKind.ACTION], legacy_action_requested=True)

    assert plan == OutputPlan.text_and_action()
    assert plan.as_strings() == ("text", "action")
    assert "text" in plan
    assert OutputKind.ACTION in plan


def test_multimodal_transport_preserves_single_string_output():
    params = MultimodalParams(outputs="action")

    assert params.outputs == ["action"]
    assert OutputPlan.from_outputs(params.outputs) == OutputPlan.action_only()


@pytest.mark.parametrize("outputs", [[], set(), ["image"], [object()]])
def test_output_plan_rejects_empty_or_unknown_outputs(outputs):
    with pytest.raises(ValueError):
        OutputPlan.from_outputs(outputs)
