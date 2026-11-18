import pytest

from hypercorn.config import Config

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _hypercorn_config_args, _validate_pd_features
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size


def test_hypercorn_config_defaults_to_none():
    args = make_argument_parser().parse_args([])

    assert args.hypercorn_config is None


def test_hypercorn_config_is_parsed():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert args.hypercorn_config == "hypercorn.toml"


def test_hypercorn_args_use_builtin_config_without_file():
    args = make_argument_parser().parse_args([])

    assert _hypercorn_config_args(args) == ["--config", "python:lightllm.server.hypercorn_config"]


def test_default_hypercorn_config_uses_256_mib_websocket_limit():
    config = Config.from_object("lightllm.server.hypercorn_config")

    assert config.keep_alive_timeout == 10
    assert config.websocket_max_message_size == get_lightllm_websocket_max_message_size()
    assert config.websocket_max_message_size == 256 * 1024 * 1024


def test_hypercorn_args_do_not_set_keep_alive_with_config():
    args = make_argument_parser().parse_args(["--hypercorn_config", "hypercorn.toml"])

    assert _hypercorn_config_args(args) == ["--config", "hypercorn.toml"]


@pytest.mark.parametrize("run_mode", ["pd_master", "prefill", "decode"])
def test_pd_nodes_reject_custom_hypercorn_config(run_mode):
    args = StartArgs(run_mode=run_mode, hypercorn_config="custom.toml")

    with pytest.raises(ValueError, match="not supported in PD mode"):
        _validate_pd_features(args)
