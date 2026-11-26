from unit_tests.common.attention.backends.step_fixture import make_backend_model

"""GDN step builders and impls, how GDN layers choose them, and the FlashQLA validator."""
from unit_tests.common.attention.backends.step_fixture import create_att_state

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gdn.selector import (
    select_gdn_prefill_backend,
    select_gdn_decode_backend,
    select_gdn_replay_decode_backend,
)
from lightllm.common.attention.backends.gdn import GDNAttConfig
from lightllm.common.attention.backends.gdn.triton import TritonGDNDecodeBackend, TritonGDNPrefillBackend
from lightllm.common.attention.backends.gdn.replay import ReplayGDNDecodeBackend
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.basemodel.infer_struct import InferStateInfo
import lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk as fla_chunk
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from lightllm.server.api_cli import make_argument_parser
import lightllm.utils.backend_validator as backend_validator


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_out", [False, True])
def test_prefill_output_storage_and_final_state_dtype(monkeypatch, cache_dtype, use_out):
    from lightllm.common.attention.backends.gdn import GDNAttConfig, GDNAttInputs
    from lightllm.common.attention.backends.gdn.triton import GDNPrefillAttState, TritonGDNPrefillBackend
    from lightllm.common.attention.kernels.gdn import causal_conv1d, fused_gdn_gating

    ssm_states = torch.zeros((1, 1, 2, 2), dtype=cache_dtype)
    final_state = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    monkeypatch.setattr(fused_gdn_gating, "fused_gdn_gating", lambda _log, a, b, _bias: (a, b))
    monkeypatch.setattr(causal_conv1d, "causal_conv1d_fn", lambda x, *args, **kwargs: x)
    core = torch.zeros((1, 1, 1, 2), dtype=cache_dtype)
    monkeypatch.setattr(
        TritonGDNPrefillBackend, "chunk_gated_delta_rule", staticmethod(lambda: lambda **kwargs: (core, final_state))
    )
    att_config = GDNAttConfig(
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=2,
        head_v_dim=2,
        conv_kernel_dim=2,
        activation="silu",
        dtype=cache_dtype,
    )
    zeros = torch.zeros((1, 1), dtype=cache_dtype)
    layer_inputs = GDNAttInputs(
        mixed_qkv=torch.zeros((1, 6), dtype=cache_dtype),
        a=zeros,
        b=zeros,
        z=torch.zeros((1, 1, 2), dtype=cache_dtype),
        conv_weight=None,
        conv_bias=None,
        A_log=None,
        dt_bias=None,
        conv_states=torch.zeros((1, 6, 1), dtype=cache_dtype),
        ssm_states=ssm_states,
    )
    index = torch.tensor([0], dtype=torch.int64)
    att_state = GDNPrefillAttState(
        conv_idx=index, ssm_idx=index, cu_seqlens=torch.tensor([0, 1], dtype=torch.int32), has_initial_state=index > 0
    )
    out = torch.full((1, 1, 2), float("nan"), dtype=cache_dtype) if use_out else None

    result = TritonGDNPrefillBackend(att_config=att_config).prefill(att_state, layer_inputs, out=out)

    assert result.data_ptr() == (out if use_out else core).data_ptr()
    assert torch.equal(result, core.squeeze(0))
    assert ssm_states.dtype == cache_dtype
    assert torch.equal(ssm_states, final_state.to(cache_dtype))


def _decode_batch(b_req_idx, *, rows=1, dynamic=False, req_to_mtp_state_index=None, hold=-1):
    from unit_tests.common.attention.backends.step_fixture import make_infer_state

    return make_infer_state(
        req_to_token_indexs=torch.zeros((8, 4), dtype=torch.int32),
        b_req_idx=b_req_idx,
        b_seq_len=torch.ones_like(b_req_idx),
        batch_size=b_req_idx.shape[0],
        max_kv_seq_len=1,
        total_token_num=b_req_idx.shape[0],
        decode_rows_per_request=rows,
        dynamic_decode_rows=dynamic,
        pad_req_idx=hold,
        b_mtp_index=torch.zeros_like(b_req_idx),
        req_to_mtp_state_index=req_to_mtp_state_index,
    )


def test_decode_step_builder_uses_request_slots_without_speculation():
    b_req_idx = torch.tensor([2, 4], dtype=torch.int32)
    att_state = create_att_state(
        backend=TritonGDNDecodeBackend(att_config=gdn_config()), infer_state=_decode_batch(b_req_idx)
    )

    assert att_state.conv_idx is b_req_idx and att_state.ssm_idx is b_req_idx
    assert att_state.cu_seqlens is None and att_state.num_accepted_tokens is None
    replay_args = create_att_state(
        backend=ReplayGDNDecodeBackend(att_config=gdn_config(replay=True)), infer_state=_decode_batch(b_req_idx)
    )
    torch.testing.assert_close(replay_args.cu_seqlens, torch.tensor([0, 1, 2], dtype=torch.int32))


def test_decode_step_builder_groups_fixed_speculative_rows():
    b_req_idx = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.int32)
    req_to_mtp_state_index = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    batch = _decode_batch(b_req_idx, rows=3, req_to_mtp_state_index=req_to_mtp_state_index)

    att_state = create_att_state(backend=TritonGDNDecodeBackend(att_config=gdn_config()), infer_state=batch)

    torch.testing.assert_close(att_state.cu_seqlens, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(att_state.conv_idx, torch.tensor([2, 4], dtype=torch.int32))
    torch.testing.assert_close(att_state.num_accepted_tokens, torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(att_state.ssm_idx, torch.tensor([[6, 7, 8], [12, 13, 14]], dtype=torch.int32))
    # 重放状态为每个请求保持一个 SSM 插槽。
    replay_args = create_att_state(
        backend=ReplayGDNDecodeBackend(att_config=gdn_config(replay=True)), infer_state=batch
    )
    torch.testing.assert_close(replay_args.ssm_idx, torch.tensor([2, 4], dtype=torch.int32))


def test_decode_step_builder_groups_dynamic_speculative_rows(monkeypatch):
    from lightllm.common.attention.kernels.gdn import mtp_state_params

    b_req_idx = torch.tensor([2, 2, 4, 4, 4], dtype=torch.int32)
    req_to_mtp_state_index = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    expected_cu_q_seq_len = torch.tensor([0, 2, 5, 5, 5, 5], dtype=torch.int32)
    expected_conv_buffer_idx = torch.tensor([2, 4, 5, 5, 5], dtype=torch.int32)
    expected_num_accepted_tokens = torch.tensor([2, 3, 1, 1, 1], dtype=torch.int32)
    build_calls = []

    def build_params(**kwargs):
        build_calls.append(kwargs)
        return expected_cu_q_seq_len, expected_conv_buffer_idx, expected_num_accepted_tokens

    monkeypatch.setattr(mtp_state_params, "build_dynamic_mtp_linear_att_state_params", build_params)
    batch = _decode_batch(b_req_idx, rows=3, dynamic=True, req_to_mtp_state_index=req_to_mtp_state_index, hold=5)

    att_state = create_att_state(backend=TritonGDNDecodeBackend(att_config=gdn_config()), infer_state=batch)

    assert att_state.cu_seqlens is expected_cu_q_seq_len
    assert att_state.conv_idx is expected_conv_buffer_idx
    assert att_state.num_accepted_tokens is expected_num_accepted_tokens
    torch.testing.assert_close(
        att_state.ssm_idx,
        torch.tensor([[6, 7, 8], [12, 13, 14], [15, 16, 17], [15, 16, 17], [15, 16, 17]], dtype=torch.int32),
    )
    assert len(build_calls) == 1
    assert build_calls[0]["b_req_idx"] is batch.b_req_idx
    assert build_calls[0]["b_mtp_index"] is batch.b_mtp_index
    assert build_calls[0]["req_to_mtp_state_index"] is req_to_mtp_state_index
    assert build_calls[0]["hold_req_id"] == 5


@pytest.mark.parametrize("role", ["main", "draft0"])
def test_decode_reads_the_acceptance_index_of_its_role(role: str):
    b_req_idx = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.int32)
    req_to_mtp_state_index = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.int32)
    model, infer_state = _speculative_decode(
        draft_step=2, b_req_idx=b_req_idx, req_to_mtp_state_index=req_to_mtp_state_index, role=role
    )
    infer_state.att_state_role = role
    infer_state.att_state_manager = model.att_state_manager
    from dataclasses import replace

    runtime = replace(TritonGDNDecodeBackend.create_runtime_config(model=model), logical_layer_ids=(3,))
    backend = TritonGDNDecodeBackend(att_config=gdn_config(), role=role, runtime_config=runtime)
    att_state = create_att_state(backend=backend, infer_state=infer_state)

    torch.testing.assert_close(att_state.num_accepted_tokens, torch.tensor([2, 3], dtype=torch.int32))


def test_triton_prefill_uses_the_in_tree_chunk_kernel():
    assert TritonGDNPrefillBackend.chunk_gated_delta_rule() is fla_chunk.chunk_gated_delta_rule


def gdn_config(**fields) -> GDNAttConfig:
    return GDNAttConfig(
        num_k_heads=fields.get("num_k_heads", 2),
        num_v_heads=fields.get("num_v_heads", 4),
        head_k_dim=fields.get("head_k_dim", 64),
        head_v_dim=fields.get("head_v_dim", 64),
        conv_kernel_dim=fields.get("conv_kernel_dim", 4),
        activation=fields.get("activation", "silu"),
        dtype=fields.get("dtype", torch.bfloat16),
        state_mtp_step=fields.get("state_mtp_step", 0),
        replay=fields.get("replay", False),
    )


def test_auto_selection_validates_flashqla(monkeypatch):
    validate_calls = []
    flashqla = ModuleType("flash_qla")
    flashqla.chunk_gated_delta_rule = lambda **kwargs: ("flashqla", kwargs)
    monkeypatch.setitem(sys.modules, "flash_qla", flashqla)

    monkeypatch.setattr(backend_validator, "validate", lambda backend_name: validate_calls.append(backend_name) or True)
    selection = select_gdn_prefill_backend(att_config=gdn_config(), model=make_backend_model(), logical_layer_ids=(0,))

    assert selection.get_name() == "flashqla_gdn_prefill"
    assert selection.chunk_gated_delta_rule()(q="q")[0] == "flashqla"
    assert validate_calls == ["flashqla"]

    selection = select_gdn_prefill_backend(
        att_config=gdn_config(), requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
    )
    assert selection.get_name() == "triton_gdn_prefill"
    assert validate_calls == ["flashqla"]


def test_auto_selection_falls_back_to_triton_when_flashqla_fails_validation(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_qla", SimpleNamespace(chunk_gated_delta_rule=None))
    monkeypatch.setattr(backend_validator, "validate", lambda backend_name: False)
    selection = select_gdn_prefill_backend(att_config=gdn_config(), model=make_backend_model(), logical_layer_ids=(0,))
    assert selection.get_name() == "triton_gdn_prefill"


@pytest.mark.parametrize("replay, decode", [(False, "triton_gdn_decode"), (True, "replay_gdn_decode")])
def test_replay_state_selects_replay_decode(monkeypatch, replay, decode):
    monkeypatch.setitem(sys.modules, "flash_qla", SimpleNamespace(chunk_gated_delta_rule=None))
    config = gdn_config(replay=replay)
    assert (
        select_gdn_prefill_backend(
            att_config=config, requested="flashqla", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == "flashqla_gdn_prefill"
    )
    select_decode = select_gdn_replay_decode_backend if replay else select_gdn_decode_backend
    assert (
        select_decode(
            att_config=config, requested="triton", model=make_backend_model(), logical_layer_ids=(0,)
        ).get_name()
        == decode
    )


def test_flashqla_validation_loads_linear_cache_config(monkeypatch):
    linear_config = SimpleNamespace(
        tp_world_size=2,
        global_linear_k_heads=4,
        global_linear_v_heads=8,
        head_linear_k_dim=8,
        head_linear_v_dim=16,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
    )
    load_calls = []
    monkeypatch.setattr(
        Qwen3NextResourceConfig,
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
    monkeypatch.setattr(fla_chunk, "chunk_gated_delta_rule", kernel)

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


def test_cli_accepts_independent_gqa_and_gdn_backends():
    args = make_argument_parser().parse_args(
        [
            "--att_backend",
            "gqa.prefill=fa3",
            "gdn.prefill=flashqla",
            "--att_backend",
            "gqa.decode=flashinfer",
            "gdn.decode=triton",
        ]
    )

    assert args.att_backend == {
        "gqa.prefill": "fa3",
        "gdn.prefill": "flashqla",
        "gqa.decode": "flashinfer",
        "gdn.decode": "triton",
    }


def _speculative_decode(*, draft_step, b_req_idx, req_to_mtp_state_index, role="main"):
    """A model of role whose GDN state keeps req_to_mtp_state_index, and a decode step of b_req_idx rows."""
    model = SimpleNamespace(
        args=SimpleNamespace(mtp_dynamic_verify=False),
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda is_draft_model: draft_step),
        att_state_role=role,
        graph_max_len_in_batch=8,
    )
    infer_state = InferStateInfo()
    infer_state.batch_size = b_req_idx.shape[0]
    infer_state.b_req_idx = b_req_idx
    infer_state.b_mtp_index = torch.zeros_like(b_req_idx)
    infer_state.req_manager = SimpleNamespace(
        HOLD_REQUEST_ID=req_to_mtp_state_index.shape[0] - 1, req_to_token_indexs=torch.zeros((8, 4), dtype=torch.int32)
    )
    request_capacity = req_to_mtp_state_index.numel()
    resources = AttentionStateManager(request_capacity=request_capacity)
    config = ConvSSMStateConfig(
        logical_layer_ids=(3,),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
    )
    for model_role in ("main", "draft0"):
        linear = GDNState(config, request_capacity=request_capacity, mtp_step=draft_step, device="meta")
        linear.mtp_state_index = (
            req_to_mtp_state_index if model_role == role else torch.zeros_like(req_to_mtp_state_index)
        )
        resources.register_state(model_role, linear)
    resources.finish_state_registration()
    model.att_state_manager = resources
    infer_state.is_prefill = False
    infer_state.b_seq_len = torch.ones_like(b_req_idx)
    return model, infer_state
