"""Utilities for speculative output retained by an API response."""


def summarize_committed_output(output_ranges, visible_output_tokens):
    committed = dict(proposal_rounds=0, output_tokens=0, accepted_draft_tokens=0, unused_proposal_rounds=0)
    for bootstrap, width, length, start in output_ranges:
        if bootstrap:
            continue
        visible = max(0, min(length, visible_output_tokens - start + 1))
        if visible:
            committed["proposal_rounds"] += 1
            committed["output_tokens"] += visible
            committed["accepted_draft_tokens"] += min(length - 1, visible)
        else:
            committed["unused_proposal_rounds"] += 1
    return committed


def trim_spec_acceptance_metrics(metrics, visible_output_tokens):
    """Clip committed output at an application stop without changing verification counts."""
    if "visible_output_tokens" not in metrics or "verification_output_ranges" not in metrics:
        return metrics
    if visible_output_tokens >= metrics["visible_output_tokens"]:
        return metrics
    result = dict(metrics)
    result.setdefault("backend_visible_output_tokens", metrics["visible_output_tokens"])
    result["visible_output_tokens"] = visible_output_tokens
    result["committed"] = summarize_committed_output(metrics["verification_output_ranges"], visible_output_tokens)
    return result
