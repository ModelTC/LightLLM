import argparse
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.cli import main
from lightllm.cli_defaults import resolve_tp
from lightllm.server.api_cli import add_cli_args
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def launch(monkeypatch):
    launch = Mock()
    monkeypatch.setitem(sys.modules, "lightllm.server.api_server", SimpleNamespace(launch_server=launch))
    return launch


@pytest.mark.parametrize(
    "arguments,legacy_arguments",
    [
        (["/models/example"], ["--model_dir", "/models/example"]),
        (
            [
                "/models/example",
                "--tp",
                "2",
                "--port",
                "9000",
                "--disable_cudagraph",
                "--llm_prefill_att_backend",
                "triton",
            ],
            [
                "--model_dir",
                "/models/example",
                "--tp",
                "2",
                "--port",
                "9000",
                "--disable_cudagraph",
                "--llm_prefill_att_backend",
                "triton",
            ],
        ),
        (["--model_dir", "/models/example"], ["--model_dir", "/models/example"]),
        (["--run_mode", "pd_master"], ["--run_mode", "pd_master"]),
        (["--run_mode", "config_server"], ["--run_mode", "config_server"]),
    ],
)
def test_serve_preserves_server_arguments(launch, arguments, legacy_arguments):
    expected = StartArgs(**vars(add_cli_args(argparse.ArgumentParser()).parse_args(legacy_arguments)))
    main(["serve", *arguments])
    launch.assert_called_once_with(expected)


def test_positional_model_uses_automatic_tp_when_not_overridden(launch, monkeypatch):
    monkeypatch.setattr("lightllm.cli.resolve_tp", lambda model_dir: 2)

    main(["serve", "/models/example"])

    args = launch.call_args.args[0]
    assert args.model_dir == "/models/example"
    assert args.tp == 2


def test_explicit_tp_skips_automatic_tp(launch, monkeypatch):
    resolver = Mock(side_effect=AssertionError("automatic TP should not run"))
    monkeypatch.setattr("lightllm.cli.resolve_tp", resolver)

    main(["serve", "/models/example", "--tp", "1"])

    assert launch.call_args.args[0].tp == 1
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "config,gpu_count,weight_gib,expected",
    [
        ({"text_config": {"num_attention_heads": 32, "num_key_value_heads": 8}}, 4, 60, 4),
        ({"num_attention_heads": 40, "num_key_value_heads": 8}, 8, 110, 8),
        ({"num_attention_heads": 32, "num_key_value_heads": 8}, 5, 140, 4),
        ({}, 4, 60, 1),
    ],
)
def test_auto_tp_respects_model_heads(tmp_path, monkeypatch, config, gpu_count, weight_gib, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (tmp_path / "model.safetensors").open("wb") as weights:
        weights.truncate(weight_gib * 1024 ** 3)
    monkeypatch.setattr("lightllm.cli_defaults._gpu_memories_mib", lambda: [40 * 1024] * gpu_count)

    assert resolve_tp(str(tmp_path)) == expected


@pytest.mark.parametrize(
    "arguments,message",
    [
        ([], "required"),
        (["serve"], "a model path is required"),
        (["serve", "/a", "--model_dir", "/b"], "specify MODEL or --model_dir"),
        (["serve", "/a", "--unknown"], "unrecognized arguments"),
        (["serve", "/a", "--tp", "bad"], "invalid int value"),
    ],
)
def test_invalid_arguments_do_not_launch(launch, capsys, arguments, message):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    launch.assert_not_called()


@pytest.mark.parametrize("option", ["--help", "--help-all"])
def test_serve_help(launch, capsys, option):
    with pytest.raises(SystemExit) as error:
        main(["serve", option])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "MODEL" in output
    assert "--tp" in output
    assert ("--llm_prefill_att_backend" in output) == (option == "--help-all")
    launch.assert_not_called()
