import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "run_mode, automatic_strategy", [("normal", "default"), ("prefill", "hrrn"), ("decode", "default")]
)
@pytest.mark.parametrize("configured_strategy", [None, "none", "default", "promote_shortest", "hrrn"])
def test_prefill_queue_strategy_resolved_before_model_loading(
    monkeypatch, run_mode, automatic_strategy, configured_strategy
):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr(f"lightllm.server.api_start.{name}", lambda args: None)

    def stop_before_model_loading(model_dir):
        raise RuntimeError("prefill queue strategy configured")

    monkeypatch.setattr("lightllm.server.api_start.has_vision_module", stop_before_model_loading)
    cli_args = ["--run_mode", run_mode]
    if configured_strategy is not None:
        cli_args.extend(["--prefill_queue_strategy", configured_strategy])
    args = StartArgs(**vars(make_argument_parser().parse_args(cli_args)))
    assert args.prefill_queue_strategy == (configured_strategy or "none")

    with pytest.raises(RuntimeError, match="prefill queue strategy configured"):
        _launch_subprocesses(args)

    expected = automatic_strategy if configured_strategy in (None, "none") else configured_strategy
    assert args.prefill_queue_strategy == expected


def test_start_args_default_prefill_queue_strategy_matches_cli():
    assert StartArgs().prefill_queue_strategy == make_argument_parser().parse_args([]).prefill_queue_strategy == "none"
