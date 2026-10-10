import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _validate_ssm_state_mode
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "mode,dtype,step,model,capacity,error",
    [
        ("native", "bfloat16", 0, "llama", 16, None),
        ("compact", "bfloat16", 2, "qwen3_5", 16, None),
        ("compact", "float32", 2, "qwen3_5", 16, None),
        ("compact", "float32", 32, "qwen3_5", 16, None),
        ("compact", "bfloat16", 2, "glm5_next", 16, None),
        ("replay", "float32", 0, "glm5_next", 8, None),
        ("replay", "bfloat16", 2, "glm5_next_text", 8, None),
        ("replay", "float32", 0, "qwen3_next", 16, None),
        ("replay", "float32", 15, "qwen3_5", 16, None),
        ("compact", "bfloat16", 0, "qwen3_5", 16, "requires MTP"),
        ("replay", "bfloat16", 2, "qwen3_5", 16, None),
        ("replay", "bfloat16", 2, "qwen3_5", 4, None),
        ("replay", "bfloat16", 0, "qwen3_5", 16, None),
        ("replay", "float32", 16, "qwen3_5", 16, "capacity"),
        ("compact", "float32", 2, "llama", 16, "require a GDN"),
        ("replay", "float32", 2, "llama", 16, "require a GDN"),
    ],
)
def test_ssm_mode_validation(monkeypatch, mode, dtype, step, model, capacity, error):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: model)
    args = make_argument_parser().parse_args(
        [
            "--ssm_state_mode",
            mode,
            "--linear_att_ssm_data_type",
            dtype,
            "--mtp_step",
            str(step),
            "--replayssm_cache_len",
            str(capacity),
        ]
    )
    if error:
        with pytest.raises(AssertionError, match=error):
            _validate_ssm_state_mode(args)
    else:
        _validate_ssm_state_mode(args)


def test_ssm_mode_defaults_and_removed_flag():
    parser = make_argument_parser()
    assert parser.parse_args([]).ssm_state_mode == StartArgs().ssm_state_mode == "native"
    assert parser.parse_args([]).replayssm_projection_mode == StartArgs().replayssm_projection_mode == "inline"
    with pytest.raises(SystemExit):
        parser.parse_args(["--enable_replayssm"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ssm_state_mode", "auto"])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"replayssm_cache_len": 2}, "power of two and at least 4"),
        ({"replayssm_cache_len": 6}, "power of two and at least 4"),
        ({"mtp_step": -1}, "mtp_step >= 0"),
        ({"linear_att_ssm_data_type": "float16"}, "FP32 or BF16"),
    ],
)
def test_programmatic_replay_config_rejected_before_model_loading(monkeypatch, overrides, error):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: "qwen3_5")
    args = StartArgs(ssm_state_mode="replay")
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(AssertionError, match=error):
        _validate_ssm_state_mode(args)


def test_replay_projection_mode_requires_replay(monkeypatch):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: "qwen3_5")
    parser = make_argument_parser()
    args = parser.parse_args(["--ssm_state_mode", "replay", "--replayssm_projection_mode", "precompute"])
    _validate_ssm_state_mode(args)
    args.ssm_state_mode = "compact"
    with pytest.raises(AssertionError, match="requires ssm_state_mode=replay"):
        _validate_ssm_state_mode(args)
