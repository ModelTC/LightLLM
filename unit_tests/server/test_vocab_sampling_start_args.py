import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("run_mode", ["normal", "prefill", "decode"])
@pytest.mark.parametrize(
    "constraint_args",
    [
        {"output_constraint_mode": "outlines"},
        {"output_constraint_mode": "xgrammar"},
        {"first_token_constraint_mode": True},
    ],
)
def test_target_candidates_reject_constraints_before_model_loading(monkeypatch, run_mode, constraint_args):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)

    def unexpected_model_loading(args):
        pytest.fail("Incompatible candidate sampling must be rejected before loading model configuration")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", unexpected_model_loading)
    args = StartArgs(run_mode=run_mode, target_vocab_topk_sampling=2, **constraint_args)

    with pytest.raises(AssertionError, match="Disable --target_vocab_topk_sampling when using output constraints"):
        _launch_subprocesses(args)


@pytest.mark.parametrize(
    "sampling_args",
    [
        {},
        {"target_vocab_topk_sampling": 2},
        {"draft_vocab_topk_sampling": 2, "output_constraint_mode": "outlines"},
        {"draft_vocab_topk_sampling": 2, "output_constraint_mode": "xgrammar"},
        {"draft_vocab_topk_sampling": 2, "first_token_constraint_mode": True},
    ],
)
def test_compatible_sampling_options_reach_model_configuration(monkeypatch, sampling_args):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)

    def stop_at_model_configuration(args):
        raise RuntimeError("model configuration reached")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", stop_at_model_configuration)

    with pytest.raises(RuntimeError, match="model configuration reached"):
        _launch_subprocesses(StartArgs(**sampling_args))
