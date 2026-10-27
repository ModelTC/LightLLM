import json

import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.api_start import _validate_and_normalize_ssm_config
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "mode,dtype,step,model,capacity,error",
    [
        ("native", "bfloat16", 0, "llama", 16, None),
        ("native", "bfloat16", 32, "llama", 16, None),
        ("compact", "bfloat16", 2, "qwen3_5", 16, None),
        ("compact", "float32", 2, "qwen3_5", 16, None),
        ("compact", "float32", 15, "qwen3_5", 16, None),
        ("compact", "float32", 16, "qwen3_5", 16, "mtp_step <= 15"),
        ("compact", "float32", 32, "qwen3_5", 16, "mtp_step <= 15"),
        ("compact", "bfloat16", 15, "glm5_next", 16, None),
        ("compact", "bfloat16", 16, "glm5_next", 16, "mtp_step <= 15"),
        ("compact", "bfloat16", 2, "glm5_next", 16, None),
        ("replay", "float32", 0, "glm5_next", 8, None),
        ("replay", "bfloat16", 2, "glm5_next_text", 8, None),
        ("replay", "float32", 0, "qwen3_next", 16, None),
        ("replay", "float32", 15, "qwen3_5", 16, None),
        ("compact", "bfloat16", 0, "qwen3_5", 16, "requires MTP"),
        ("replay", "bfloat16", 2, "qwen3_5", 16, None),
        ("replay", "bfloat16", 2, "qwen3_5", 4, None),
        ("replay", "bfloat16", 0, "qwen3_5", 16, None),
        ("replay", "float32", 4, "qwen3_5", 4, "capacity"),
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
            "--ssm_replay_config",
            json.dumps({"cache_len": capacity} if mode == "replay" else {}),
        ]
    )
    if error:
        with pytest.raises(AssertionError, match=error):
            _validate_and_normalize_ssm_config(args)
    else:
        _validate_and_normalize_ssm_config(args)


@pytest.mark.parametrize("model", ["qwen3_5", "glm5_next"])
@pytest.mark.parametrize("mode", ["native", "compact", "replay"])
@pytest.mark.parametrize("step", [15, 16])
def test_gdn_kda_mtp_width_limit(monkeypatch, model, mode, step):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: model)
    args = StartArgs(ssm_state_mode=mode, mtp_step=step)
    if step <= 15:
        _validate_and_normalize_ssm_config(args)
    else:
        with pytest.raises(AssertionError, match="GDN/KDA requires mtp_step <= 15"):
            _validate_and_normalize_ssm_config(args)


@pytest.mark.parametrize("model", ["qwen3_5", "glm5_next"])
@pytest.mark.parametrize("compute_mode", ["inline", "precompute", "parallel"])
@pytest.mark.parametrize("cache_len", [32, 64])
def test_replay_large_cache_does_not_bypass_mtp_width_limit(monkeypatch, model, compute_mode, cache_len):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: model)
    args = StartArgs(
        ssm_state_mode="replay",
        ssm_replay_config={"cache_len": cache_len, "compute_mode": compute_mode},
        linear_att_ssm_data_type="bfloat16",
        mtp_step=16,
    )
    with pytest.raises(AssertionError, match="GDN/KDA requires mtp_step <= 15"):
        _validate_and_normalize_ssm_config(args)


def test_ssm_mode_defaults_and_removed_flag():
    parser = make_argument_parser()
    assert parser.parse_args([]).ssm_state_mode == StartArgs().ssm_state_mode == "native"
    assert parser.parse_args([]).ssm_replay_config == StartArgs().ssm_replay_config == {}
    with pytest.raises(SystemExit):
        parser.parse_args(["--enable_replayssm"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ssm_state_mode", "auto"])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"ssm_replay_config": {"cache_len": 2}}, "power of two and at least 4"),
        ({"ssm_replay_config": {"cache_len": 6}}, "power of two and at least 4"),
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
        _validate_and_normalize_ssm_config(args)


def test_replay_compute_mode_requires_replay(monkeypatch):
    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: "qwen3_5")
    parser = make_argument_parser()
    args = parser.parse_args(["--ssm_state_mode", "replay", "--ssm_replay_config", '{"compute_mode":"precompute"}'])
    _validate_and_normalize_ssm_config(args)
    args.ssm_state_mode = "compact"
    with pytest.raises(AssertionError, match="requires ssm_state_mode=replay"):
        _validate_and_normalize_ssm_config(args)


@pytest.mark.parametrize("fold", ["raw", "derived"])
def test_parallel_replay_configuration(monkeypatch, fold):
    from lightllm.server.core.objs.start_args_type import StartArgs

    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: "qwen3_5")
    args = StartArgs(
        ssm_state_mode="replay",
        ssm_replay_config={"compute_mode": "parallel", "fold_method": fold},
        mtp_step=4,
        mtp_mode="eagle_with_att",
        linear_att_ssm_data_type="bfloat16",
    )
    _validate_and_normalize_ssm_config(args)
    args.mtp_step = 0
    with pytest.raises(AssertionError, match="requires MTP"):
        _validate_and_normalize_ssm_config(args)


def test_derived_replay_requires_parallel_configuration():
    from lightllm.server.core.objs.start_args_type import StartArgs

    args = StartArgs(ssm_state_mode="replay", ssm_replay_config={"fold_method": "derived"})
    with pytest.raises(AssertionError, match="requires parallel"):
        _validate_and_normalize_ssm_config(args)


@pytest.mark.parametrize(
    "config,error",
    [
        ([], "JSON object"),
        (None, "JSON object"),
        ({"projection_mode": "inline"}, "Unknown"),
        ({"cache_len": True}, "capacity"),
        ({"cache_len": 16.0}, "capacity"),
        ({"cache_len": "16"}, "capacity"),
        ({"cache_len": 128}, "capacity"),
        ({"compute_mode": "auto"}, "compute_mode"),
        ({"fold_method": "auto"}, "fold_method"),
    ],
)
def test_invalid_replay_config(config, error):
    args = StartArgs(ssm_state_mode="replay", ssm_replay_config=config)
    with pytest.raises(AssertionError, match=error):
        _validate_and_normalize_ssm_config(args)


@pytest.mark.parametrize("mode", ["native", "compact"])
def test_replay_config_rejects_other_modes(mode):
    args = StartArgs(ssm_state_mode=mode, ssm_replay_config={"cache_len": 16})
    with pytest.raises(AssertionError, match="requires ssm_state_mode=replay"):
        _validate_and_normalize_ssm_config(args)


def test_partial_replay_config_defaults_and_serialization(monkeypatch):
    from dataclasses import asdict, fields

    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: "qwen3_5")
    args = make_argument_parser().parse_args(
        ["--ssm_state_mode", "replay", "--ssm_replay_config", '{"compute_mode":"precompute"}']
    )
    _validate_and_normalize_ssm_config(args)
    assert args.ssm_replay_config == {"cache_len": 16, "compute_mode": "precompute", "fold_method": "raw"}
    original = StartArgs(ssm_state_mode="replay", ssm_replay_config=args.ssm_replay_config)
    serialized = json.loads(json.dumps(asdict(original)))
    init_fields = {field.name for field in fields(StartArgs) if field.init}
    restored = StartArgs(**{name: value for name, value in serialized.items() if name in init_fields})
    _validate_and_normalize_ssm_config(restored)
    assert restored.ssm_replay_config == args.ssm_replay_config
    assert StartArgs().ssm_replay_config == {}


@pytest.mark.parametrize(
    "option,value",
    [
        ("--replayssm_cache_len", "16"),
        ("--replayssm_projection_mode", "inline"),
        ("--replayssm_fold_method", "raw"),
        ("--ssm_replay_config", "{invalid}"),
    ],
)
def test_removed_flags_and_malformed_json(option, value):
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args([option, value])
