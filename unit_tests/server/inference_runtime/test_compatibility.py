from types import SimpleNamespace

import pytest

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.inference_runtime.compatibility import configure_vla_runtime


def _args() -> StartArgs:
    args = StartArgs()
    args.run_mode = "normal"
    args.tp = 2
    args.dp = 1
    args.nnodes = 1
    args.max_req_total_len = None
    args.vla_max_prefix_tokens = 128
    return args


@pytest.fixture
def fake_pi_config(monkeypatch):
    config = SimpleNamespace(action_horizon=50)
    monkeypatch.setattr(
        Pi0VLAConfig,
        "from_start_args",
        classmethod(lambda cls, args: config),
    )
    monkeypatch.setattr(
        "lightllm.server.inference_runtime.compatibility.mp.set_start_method",
        lambda *args, **kwargs: None,
    )
    return config


def test_ordinary_checkpoint_does_not_mount_action_runtime():
    args = _args()

    assert not configure_vla_runtime(args, model_type="qwen2")
    assert not args.enable_vla
    assert args.run_mode == "normal"


def test_pi_checkpoint_uses_normal_backend_with_optional_runtime(fake_pi_config):
    args = _args()

    assert configure_vla_runtime(args, model_type="pi0")
    assert args.enable_vla
    assert args.run_mode == "normal"
    assert args.action_tp == args.tp
    assert args.action_gpu_ids == [0, 1]
    assert args.max_req_total_len == 179


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("run_mode", "prefill", "PD-separated"),
        ("dp", 2, "DP"),
        ("nnodes", 2, "multi-node TP"),
        ("enable_cpu_cache", True, "multi-level KV cache"),
        ("output_constraint_mode", "xgrammar", "constraint decoding"),
        (
            "enable_prefill_microbatch_overlap",
            True,
            "prefill microbatch overlap",
        ),
        ("enable_prefill_decode_mixed", True, "mixed prefill/decode batching"),
    ],
)
def test_unvalidated_action_combinations_fail_fast(fake_pi_config, attribute, value, message):
    args = _args()
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        configure_vla_runtime(args, model_type="pi05")
