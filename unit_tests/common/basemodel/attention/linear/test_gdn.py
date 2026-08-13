import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.attention.linear.gdn as gdn
import lightllm.common.basemodel.attention.create_linear_utils as linear_create_utils
import lightllm.common.basemodel.triton_kernel.linear_att.fla.ops as fla_ops
from lightllm.common.basemodel.attention.linear.flashqla import FlashQlaLinearAttBackend
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.server.api_cli import make_argument_parser
import lightllm.utils.backend_validator as backend_validator


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_prefill_casts_final_state_to_cache_dtype(monkeypatch, cache_dtype):
    ssm_states = torch.zeros((1, 1, 2, 2), dtype=cache_dtype)
    final_state = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(gdn, "fused_gdn_gating", lambda _log, a, b, _bias: (a, b))
    monkeypatch.setattr(gdn, "causal_conv1d_fn", lambda mixed, *args, **kwargs: mixed)
    qkv = torch.zeros((1, 3), dtype=cache_dtype)
    q = torch.zeros((1, 1, 1, 1), dtype=cache_dtype)
    backend = SimpleNamespace(
        mtp_step=0,
        activation="silu",
        ssm_state_dtype=cache_dtype,
        _rearrange_mixed_qkv=lambda mixed: (q, q, q),
        prefill_kernel=lambda *args, **kwargs: (None, final_state),
    )
    state = gdn.LinearAttPrefillAttState(
        backend=backend,
        infer_state=SimpleNamespace(
            b1_cu_q_seq_len=torch.tensor([0, 1], dtype=torch.int32),
            b_ready_cache_len=0,
        ),
    )
    state.b_conv_buffer_idx = torch.tensor([0], dtype=torch.int64)
    state.b_ssm_buffer_idx = torch.tensor([0], dtype=torch.int64)
    layer_weight = SimpleNamespace(
        linear_A_log=SimpleNamespace(weight=None),
        linear_dt_bias=SimpleNamespace(weight=None),
        linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=None), bias=None),
    )

    state._gdn_prefill_kernel(
        qkv,
        torch.zeros((1, 3), dtype=cache_dtype),
        ssm_states,
        torch.zeros((1, 1), dtype=cache_dtype),
        torch.zeros((1, 1), dtype=cache_dtype),
        state.infer_state,
        layer_weight,
    )

    assert ssm_states.dtype == cache_dtype
    assert torch.equal(ssm_states, final_state.to(cache_dtype))


def test_linear_backend_is_abstract_and_triton_supplies_chunk_kernel():
    assert inspect.isabstract(gdn.LinearAttBackend)
    backend = object.__new__(TritonLinearAttBackend)
    assert backend.get_prefill_kernel() is fla_ops.chunk_gated_delta_rule


@pytest.fixture
def auto_linear_backend_args(monkeypatch):
    monkeypatch.setattr(
        linear_create_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            llm_prefill_att_backend=["fa3"],
            llm_decode_att_backend=["flashinfer"],
        ),
    )


def test_missing_linear_backend_arg_uses_auto_selection(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = lambda **kwargs: ("flashqla", kwargs)
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name: validate_calls.append(name) or True,
    )

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)

    assert backend_class is FlashQlaLinearAttBackend
    backend = object.__new__(backend_class)
    assert backend.get_prefill_kernel()(q="q")[0] == "flashqla"
    assert validate_calls == ["flashqla"]


def test_flashqla_validation_loads_linear_cache_config(monkeypatch):
    linear_config = SimpleNamespace(
        num_linear_k_heads=2,
        num_linear_v_heads=4,
        head_linear_k_dim=8,
        head_linear_v_dim=16,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    load_calls = []
    monkeypatch.setattr(
        LinearAttCacheConfig,
        "load_from_args",
        staticmethod(lambda: load_calls.append(True) or linear_config),
    )

    kernel_calls = []

    def kernel(**kwargs):
        kernel_calls.append(kwargs)
        return kwargs["q"], kwargs["initial_state"]

    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = kernel
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)
    monkeypatch.setattr(fla_ops, "chunk_gated_delta_rule", kernel)

    original_randn = torch.randn
    original_rand = torch.rand
    original_tensor = torch.tensor

    def cpu_randn(*args, **kwargs):
        kwargs.pop("device", None)
        return original_randn(*args, **kwargs)

    def cpu_rand(*args, **kwargs):
        kwargs.pop("device", None)
        return original_rand(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backend_validator.torch, "randn", cpu_randn)
    monkeypatch.setattr(backend_validator.torch, "rand", cpu_rand)
    monkeypatch.setattr(backend_validator.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(backend_validator.torch.cuda, "synchronize", lambda: None)

    success, error = backend_validator._validate_flashqla()

    assert success is True
    assert error is None
    assert load_calls == [True]
    assert len(kernel_calls) == 2
    assert kernel_calls[0]["q"].shape == (1, 64, 2, 8)
    assert kernel_calls[0]["v"].shape == (1, 64, 4, 16)
    assert kernel_calls[0]["q"].dtype == torch.bfloat16
    assert kernel_calls[0]["initial_state"].shape == (1, 4, 8, 16)
    assert kernel_calls[0]["initial_state"].dtype == torch.float32


def test_explicit_linear_backend_arg_skips_auto_selection(monkeypatch):
    monkeypatch.setattr(
        linear_create_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            llm_prefill_att_backend=["fa3", "triton"],
            llm_decode_att_backend=["flashinfer", "triton"],
        ),
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda *args: pytest.fail("an explicit linear backend must not be auto-validated"),
    )

    prefill_backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)
    decode_backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(index=1)

    assert prefill_backend_class is TritonLinearAttBackend
    assert decode_backend_class is TritonLinearAttBackend


def test_cli_accepts_linear_backend_at_index_one():
    args = make_argument_parser().parse_args(
        [
            "--llm_prefill_att_backend",
            "fa3",
            "flashqla",
            "--llm_decode_att_backend",
            "flashinfer",
            "triton",
        ]
    )

    assert args.llm_prefill_att_backend == ["fa3", "flashqla"]
    assert args.llm_decode_att_backend == ["flashinfer", "triton"]


@pytest.mark.parametrize(
    "get_backend_class",
    [
        linear_create_utils.get_qwen35_linear_prefill_att_backend_class,
        linear_create_utils.get_qwen35_linear_decode_att_backend_class,
    ],
)
def test_linear_backend_rejects_non_linear_index(auto_linear_backend_args, get_backend_class):
    with pytest.raises(AssertionError, match="index must be 1"):
        get_backend_class(index=0)


def test_missing_linear_decode_backend_arg_defaults_to_triton(auto_linear_backend_args):
    backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(index=1)

    assert backend_class is TritonLinearAttBackend


def test_gdn_decode_backend_uses_priority_list(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    optimized_backend = type("OptimizedLinearDecodeAttBackend", (), {})
    monkeypatch.setattr(
        linear_create_utils,
        "linear_decode_att_backend_classes",
        {"optimized": optimized_backend, "triton": TritonLinearAttBackend},
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name: validate_calls.append(name) or True,
    )

    backend_class = linear_create_utils.get_qwen35_linear_decode_att_backend_class(
        index=1, priority_list=("optimized", "triton")
    )

    assert backend_class is optimized_backend
    assert validate_calls == ["optimized"]


def test_gdn_prefill_backend_falls_back_when_flashqla_validation_fails(monkeypatch, auto_linear_backend_args):
    monkeypatch.setattr(linear_create_utils, "validate", lambda *args: False)

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(index=1)

    assert backend_class is TritonLinearAttBackend


def test_gdn_prefill_backend_tries_candidates_in_order(monkeypatch, auto_linear_backend_args):
    validate_calls = []
    flashqla2_backend = type("FlashQla2LinearAttBackend", (), {})
    flashqla3_backend = type("FlashQla3LinearAttBackend", (), {})
    monkeypatch.setattr(
        linear_create_utils,
        "linear_prefill_att_backend_classes",
        {"flashqla2": flashqla2_backend, "flashqla3": flashqla3_backend},
    )
    monkeypatch.setattr(
        linear_create_utils,
        "validate",
        lambda name, *args: validate_calls.append(name) or name == "flashqla3",
    )

    backend_class = linear_create_utils.get_qwen35_linear_prefill_att_backend_class(
        index=1, priority_list=("flashqla2", "flashqla3")
    )

    assert backend_class is flashqla3_backend
    assert validate_calls == ["flashqla2", "flashqla3"]
