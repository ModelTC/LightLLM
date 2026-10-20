import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _auto_set_prefill_queue_strategy
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "run_mode, automatic_strategy",
    [("normal", "default"), ("prefill", "hrrn"), ("decode", "default"), ("visual_only", "default")],
)
@pytest.mark.parametrize("configured_strategy", [None, "auto", "default", "promote_shortest", "hrrn"])
def test_auto_set_prefill_queue_strategy(run_mode, automatic_strategy, configured_strategy):
    cli_args = ["--run_mode", run_mode]
    if configured_strategy is not None:
        cli_args.extend(["--prefill_queue_strategy", configured_strategy])
    args = StartArgs(**vars(make_argument_parser().parse_args(cli_args)))
    assert args.prefill_queue_strategy == (configured_strategy or "auto")
    expected = automatic_strategy if configured_strategy in (None, "auto") else configured_strategy
    _auto_set_prefill_queue_strategy(args)
    assert args.prefill_queue_strategy == expected


def test_start_args_default_prefill_queue_strategy_matches_cli():
    assert StartArgs().prefill_queue_strategy == make_argument_parser().parse_args([]).prefill_queue_strategy == "auto"
    cli_option = next(action for action in make_argument_parser()._actions if action.dest == "prefill_queue_strategy")
    assert set(cli_option.choices) == set(StartArgs.__dataclass_fields__["prefill_queue_strategy"].metadata["choices"])
