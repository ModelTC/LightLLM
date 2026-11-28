import pytest

from lightllm.server.router.model_infer.mtp_speculative.acceptance_metrics import SpecAcceptanceMetrics


def test_fixed_width_keeps_initialization_separate_and_conserves_counts():
    metrics = SpecAcceptanceMetrics()
    for accepted in [1, 6, 3, 1]:
        metrics.record(accepted, 6)
    result = metrics.export()
    assert result["bootstrap"]["rounds"] == 1
    assert result["bootstrap"]["accepted_draft_tokens"] == 0
    proposal = result["proposals"]
    assert proposal["rounds"] == 3
    assert proposal["accepted_draft_tokens"] == 7
    assert proposal["proposed_draft_tokens"] == 15
    assert proposal["positions"] == [
        dict(position=1, proposed=3, reached=3, accepted=2),
        dict(position=2, proposed=3, reached=2, accepted=2),
        dict(position=3, proposed=3, reached=2, accepted=1),
        dict(position=4, proposed=3, reached=1, accepted=1),
        dict(position=5, proposed=3, reached=1, accepted=1),
    ]
    assert sum(p["accepted"] for p in proposal["positions"]) == proposal["accepted_draft_tokens"]


def test_varying_widths_do_not_count_unproposed_positions_as_rejections():
    metrics = SpecAcceptanceMetrics()
    for accepted, width in [(1, 1), (2, 2), (4, 4), (1, 4)]:
        metrics.record(accepted, width)
    positions = metrics.export()["proposals"]["positions"]
    assert positions == [
        dict(position=1, proposed=3, reached=3, accepted=2),
        dict(position=2, proposed=2, reached=1, accepted=1),
        dict(position=3, proposed=2, reached=1, accepted=1),
    ]


@pytest.mark.parametrize("accepted,width", [(0, 1), (2, 1), (1, 0)])
def test_invalid_round_is_rejected(accepted, width):
    with pytest.raises(ValueError):
        SpecAcceptanceMetrics().record(accepted, width)


def test_new_request_has_no_previous_slot_history_and_exports_are_snapshots():
    first = SpecAcceptanceMetrics()
    first.record(1, 6)
    saved = first.export()
    first.record(6, 6)
    assert saved["proposals"]["rounds"] == 0
    second = SpecAcceptanceMetrics().export()
    assert second["bootstrap"]["rounds"] == second["proposals"]["rounds"] == 0


def test_pause_recovery_keeps_history_and_separates_the_next_initialization():
    metrics = SpecAcceptanceMetrics()
    metrics.record(1, 6)
    metrics.record(6, 6)
    metrics.mark_bootstrap()
    metrics.record(2, 6)
    metrics.record(3, 6)
    result = metrics.export()
    assert result["bootstrap"]["rounds"] == result["proposals"]["rounds"] == 2
    assert result["bootstrap"]["accepted_draft_tokens"] == 1
    assert result["proposals"]["accepted_draft_tokens"] == 7


@pytest.mark.parametrize("with_final_snapshot", [False, True])
def test_metrics_only_final_metadata_round_trip(monkeypatch, with_final_snapshot):
    import asyncio
    from types import SimpleNamespace

    import numpy as np
    from lightllm.server.core.objs import token_metadata
    from lightllm.server.core.objs.req import Req

    class Shm:
        def __init__(self, size):
            self.buf = bytearray(size)

        def close(self):
            pass

        def unlink(self):
            pass

    buffers = {}

    def create(name, size, force_mode):
        if force_mode == "create":
            buffers[name] = Shm(size)
        return buffers[name]

    monkeypatch.setattr(token_metadata, "create_or_link_shm", create)
    monkeypatch.setattr(
        "lightllm.server.core.objs.req.get_env_start_args",
        lambda: SimpleNamespace(enable_spec_acceptance_metrics=True),
    )
    req = SimpleNamespace(
        index_in_shm_mem=11,
        request_id=77,
        input_len=1,
        shm_prompt_ids=SimpleNamespace(arr=np.array([123])),
        sample_params=SimpleNamespace(prompt_logprobs=-1),
        shm_infer_released=True,
    )
    metrics = SpecAcceptanceMetrics()
    metrics.record(1, 6)
    metrics.record(4, 6)
    final_counters = dict(mtp_accepted_token_num=3, mtp_verify_token_num=12, mtp_verify_step_num=2)
    exported = metrics.export(native_final_counters=final_counters if with_final_snapshot else None)
    token_metadata.ReqFinalTokenMetadata(req).save(spec_acceptance_metrics=exported)
    stale = dict(mtp_accepted_token_num=0, mtp_verify_token_num=6, mtp_verify_step_num=1)
    metadata = dict(stale) if with_final_snapshot else {}
    asyncio.run(Req.merge_final_token_metadata(req, metadata, None))
    if with_final_snapshot:
        exported["http_counter_snapshot"] = stale
        assert metadata == {"spec_acceptance_metrics": exported, **final_counters}
    else:
        assert metadata == {"spec_acceptance_metrics": exported}


def test_output_budget_truncation_keeps_unconsumed_work_out_of_committed_gain():
    metrics = SpecAcceptanceMetrics()
    for length, end in [(1, 2), (6, 8), (6, 14), (4, 18)]:
        metrics.record(length, 6, output_end=end)
    result = metrics.export(visible_output_tokens=10, inferred_output_tokens=18)
    assert result["proposals"]["rounds"] == 3
    assert result["proposals"]["accepted_draft_tokens"] == 13
    assert result["committed"] == dict(
        proposal_rounds=2, output_tokens=8, accepted_draft_tokens=7, unused_proposal_rounds=1
    )


def test_cli_flag_and_pd_scope():
    from argparse import ArgumentParser
    from lightllm.server.api_cli import add_cli_args
    from lightllm.server.api_start import _validate_pd_features
    from lightllm.server.core.objs.start_args_type import StartArgs

    parser = ArgumentParser()
    add_cli_args(parser)
    assert not parser.parse_args([]).enable_spec_acceptance_metrics
    assert parser.parse_args(["--enable_spec_acceptance_metrics"]).enable_spec_acceptance_metrics
    with pytest.raises(ValueError, match="speculative acceptance metrics"):
        _validate_pd_features(StartArgs(run_mode="pd_master", enable_spec_acceptance_metrics=True))
