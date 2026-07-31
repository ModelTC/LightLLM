from lightllm.server.api_cli import make_argument_parser


def test_hypercorn_config_defaults_to_none():
    args = make_argument_parser().parse_args([])

    assert args.hypercorn_config is None


def test_hypercorn_config_is_parsed():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert args.hypercorn_config == "hypercorn.toml"
