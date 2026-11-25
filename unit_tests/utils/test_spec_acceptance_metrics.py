import pytest

from lightllm.utils.spec_acceptance_metrics import trim_spec_acceptance_metrics


@pytest.mark.parametrize(
    "visible,output,accepted,rounds,unused", [(1, 0, 0, 0, 1), (2, 0, 0, 0, 1), (3, 1, 1, 1, 0), (4, 2, 2, 1, 0)]
)
def test_application_stop_clips_committed_work_without_changing_histograms(visible, output, accepted, rounds, unused):
    original = dict(
        visible_output_tokens=5,
        inferred_output_tokens=6,
        verification_output_ranges=[[True, 6, 1, 2], [False, 6, 4, 3]],
        proposals={"rounds": 1, "accepted_draft_tokens": 3},
        native_final_counters={"mtp_accepted_token_num": 3},
    )
    result = trim_spec_acceptance_metrics(original, visible)
    assert result["visible_output_tokens"] == visible
    assert result["backend_visible_output_tokens"] == 5
    assert result["inferred_output_tokens"] == 6
    assert result["committed"] == dict(
        proposal_rounds=rounds, output_tokens=output, accepted_draft_tokens=accepted, unused_proposal_rounds=unused
    )
    assert result["proposals"] == original["proposals"]
    assert result["native_final_counters"] == original["native_final_counters"]
    assert original["visible_output_tokens"] == 5 and "committed" not in original
    again = trim_spec_acceptance_metrics(result, 1)
    assert again["backend_visible_output_tokens"] == 5


def test_complete_or_legacy_metadata_is_unchanged():
    legacy = {"schema_version": 2}
    assert trim_spec_acceptance_metrics(legacy, 1) is legacy
    complete = dict(visible_output_tokens=2, verification_output_ranges=[])
    assert trim_spec_acceptance_metrics(complete, 2) is complete
