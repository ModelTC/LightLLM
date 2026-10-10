import pytest

from lightllm.server.api_start import _launch_subprocesses, pd_master_start
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("run_mode", ["normal", "prefill", "decode"])
@pytest.mark.parametrize(
    "constraint_args",
    [
        {},
        {"output_constraint_mode": "xgrammar"},
    ],
)
def test_target_candidates_reject_constraints_before_model_configuration(monkeypatch, run_mode, constraint_args):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)

    def unexpected_initialization(args):
        pytest.fail("incompatible options must be rejected before initialization")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", unexpected_initialization)
    with pytest.raises(AssertionError, match="--target_vocab_topk_sampling"):
        _launch_subprocesses(StartArgs(run_mode=run_mode, target_vocab_topk_sampling=2, **constraint_args))


@pytest.mark.parametrize("run_mode", ["normal", "prefill", "decode", "pd_master"])
@pytest.mark.parametrize(
    "sampling_args",
    [
        {},
        {"target_vocab_topk_sampling": 2, "output_constraint_mode": "none"},
        {"draft_vocab_topk_sampling": 2},
        {"diverse_mode": True, "output_constraint_mode": "xgrammar"},
        {"use_reward_model": True, "output_constraint_mode": "xgrammar"},
        {
            "dp": 2,
            "mtp_mode": "vanilla_no_att",
            "enable_prefill_microbatch_overlap": True,
            "enable_decode_microbatch_overlap": True,
            "output_constraint_mode": "xgrammar",
        },
    ],
)
def test_compatible_sampling_options_reach_model_configuration(monkeypatch, run_mode, sampling_args):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)

    def stop_at_model_configuration(args):
        raise RuntimeError("model configuration reached")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", stop_at_model_configuration)
    start = pd_master_start if run_mode == "pd_master" else _launch_subprocesses

    with pytest.raises(RuntimeError, match="model configuration reached"):
        start(StartArgs(run_mode=run_mode, **sampling_args))
