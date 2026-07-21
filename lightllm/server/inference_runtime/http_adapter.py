from __future__ import annotations

from .output_plan import OutputPlan


def resolve_request_output_plan(
    multimodal_params,
    sampling_params,
    *,
    action_runtime_enabled: bool,
) -> OutputPlan:
    action_payload = getattr(multimodal_params, "action", None)
    plan = OutputPlan.resolve(
        getattr(multimodal_params, "outputs", None),
        legacy_action_requested=action_payload is not None,
    )
    if plan.wants_action and action_payload is None:
        raise ValueError("the action output branch requires an action payload")
    if plan.wants_action and not action_runtime_enabled:
        raise ValueError("the action output branch is not enabled for this server")
    if plan.wants_action and (
        int(getattr(sampling_params, "n", 1)) != 1 or int(getattr(sampling_params, "best_of", 1)) != 1
    ):
        raise ValueError("action output requires n=1 and best_of=1")
    if plan.wants_action and not plan.wants_text and int(getattr(sampling_params, "max_new_tokens", 1)) != 1:
        raise ValueError("action-only output requires max_new_tokens=1")
    return plan
