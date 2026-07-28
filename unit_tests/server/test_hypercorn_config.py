from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _get_hypercorn_config_args
from lightllm.server.core.objs.start_args_type import StartArgs


def test_hypercorn_config_defaults_to_none():
    args = make_argument_parser().parse_args([])

    assert args.hypercorn_config is None
    assert _get_hypercorn_config_args(StartArgs()) == []


def test_hypercorn_config_is_forwarded():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert args.hypercorn_config == "hypercorn.toml"
    assert _get_hypercorn_config_args(StartArgs(hypercorn_config=args.hypercorn_config)) == [
        "--config",
        "hypercorn.toml",
    ]
