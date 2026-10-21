from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _hypercorn_config_args


def test_hypercorn_config_defaults_to_none():
    args = make_argument_parser().parse_args([])

    assert args.hypercorn_config is None


def test_hypercorn_config_is_parsed():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert args.hypercorn_config == "hypercorn.toml"


def test_hypercorn_args_use_default_keep_alive_without_config():
    args = make_argument_parser().parse_args([])

    assert _hypercorn_config_args(args) == ["--keep-alive", "10"]


def test_hypercorn_args_do_not_set_keep_alive_with_config():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert _hypercorn_config_args(args) == ["--config", "hypercorn.toml"]
