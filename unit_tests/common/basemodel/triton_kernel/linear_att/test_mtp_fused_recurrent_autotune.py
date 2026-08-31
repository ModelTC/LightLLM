from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.linear import gdn as gdn_module
from lightllm.common.basemodel.triton_kernel.linear_att import causal_conv1d_mtp
from lightllm.common.basemodel.triton_kernel.linear_att import mtp_fused_recurrent as mtp_recurrent
from lightllm.common.triton_utils.autotuner import Autotuner


def _layout_tensors(total_tokens=8, K=64, V=128):
    H, HV = 1, 2
    q = torch.empty((1, total_tokens, H, K), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, total_tokens, HV, V), dtype=torch.bfloat16)
    return q, k, v


def test_v1_config_space_and_fallback():
    configs = mtp_recurrent._get_mtp_fused_recurrent_configs()
    assert len(configs) == 60
    assert len({tuple(sorted(config.items())) for config in configs}) == len(configs)
    assert {config["BV"] for config in configs} == {4, 8, 16, 32, 64}
    assert {config["num_warps"] for config in configs} == {1, 2, 4, 8}
    assert {config["num_stages"] for config in configs} == {1, 2, 3}
    assert mtp_recurrent._default_mtp_fused_recurrent_config(128) == {
        "BV": 8,
        "num_warps": 1,
        "num_stages": 3,
    }


def test_key_separates_fixed_and_dynamic_specializations():
    q, _, v = _layout_tensors()
    initial_state = torch.empty((16, 2, 64, 128), dtype=torch.bfloat16)
    cu_seqlens = torch.empty(3, dtype=torch.int64)

    dynamic_key = mtp_recurrent._get_mtp_fused_recurrent_static_key(q, v, initial_state, fixed_seq_len=0)
    fixed_key = mtp_recurrent._get_mtp_fused_recurrent_static_key(q, v, initial_state, fixed_seq_len=4)

    assert dynamic_key["fixed_seq_len"] == 0
    assert fixed_key["fixed_seq_len"] == 4
    assert dynamic_key != fixed_key
    assert mtp_recurrent._get_mtp_fused_recurrent_run_key(q, cu_seqlens) == 2 * 10 ** 9 + 8


def test_tuner_declares_mutated_state():
    assert mtp_recurrent._mtp_fused_recurrent_gated_delta_rule_autotuned.mutates_args == ["initial_state"]


def test_gdn_cuda_graph_warmup_scopes_v1_autotune_to_first_layer_microbatch(monkeypatch):
    Autotuner.end_autotune_warmup()
    calls = []
    q, k, v = _layout_tensors()
    backend = SimpleNamespace(
        activation="silu",
        conv_kernel_dim=4,
        model=SimpleNamespace(
            is_mtp_draft_model=False,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 3),
        ),
        uses_dynamic_spec_verify_layout=lambda: False,
        _rearrange_mixed_qkv=lambda _, decode: (q, k, v),
    )
    state = gdn_module.LinearAttDecodeAttState(backend=backend)
    state.b1_mtp_cu_q_seq_len = torch.tensor([0, 4, 8], dtype=torch.int32)
    state.b_conv_buffer_idx = torch.tensor([0, 1], dtype=torch.int32)
    state.b_ssm_buffer_idx = torch.zeros((2, 4), dtype=torch.int32)
    state.b_num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    infer_state = SimpleNamespace(is_cuda_graph=True, microbatch_index=0)
    layer_weight = SimpleNamespace(
        layer_num_=0,
        linear_conv1d=SimpleNamespace(mm_param=SimpleNamespace(weight=torch.empty(1)), bias=None),
        linear_A_log=SimpleNamespace(weight=torch.empty(1)),
        linear_dt_bias=SimpleNamespace(weight=torch.empty(1)),
    )

    monkeypatch.setattr(gdn_module, "get_triton_autotune_level", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(causal_conv1d_mtp, "causal_conv1d_update", lambda mixed_qkv, *args, **kwargs: mixed_qkv)

    def fake_recurrent(**kwargs):
        calls.append(Autotuner.is_autotune_warmup())
        return kwargs["q"], kwargs["initial_state"]

    monkeypatch.setattr(gdn_module, "mtp_fused_recurrent_gated_delta_rule", fake_recurrent)

    args = (
        torch.empty((8, 1)),
        torch.empty(1),
        torch.empty(1),
        torch.empty((8, 2)),
        torch.empty((8, 2)),
        infer_state,
        layer_weight,
    )
    state._gdn_mtp_kernel(*args)
    layer_weight.layer_num_ = 1
    state._gdn_mtp_kernel(*args)

    assert calls == [True, False]
    assert not Autotuner.is_autotune_warmup()


def _make_fixed_case():
    torch.manual_seed(11)
    batch, seq_len, H, HV, K, V = 2, 4, 1, 2, 64, 128
    total_tokens = batch * seq_len
    q = torch.randn((1, total_tokens, H, K), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((1, total_tokens, HV, V), device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn((10, HV, K, V), device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int64) * seq_len
    read_indices = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], device="cuda", dtype=torch.int32)
    write_indices = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device="cuda", dtype=torch.int32)
    accepted = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a_raw = torch.randn((total_tokens, HV), device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn_like(a_raw)
    return (
        q,
        k,
        v,
        initial_state,
        cu_seqlens,
        read_indices,
        write_indices,
        accepted,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
    )


def _launch(case, run_config, *, fixed_seq_len):
    (
        q,
        k,
        v,
        initial_state,
        cu_seqlens,
        read_indices,
        write_indices,
        accepted,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
    ) = case
    state = initial_state.clone()
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "initial_state": state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": read_indices,
        "ssm_state_write_indices": write_indices,
        "num_accepted_tokens": accepted,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "fixed_seq_len": fixed_seq_len,
        "run_config": run_config,
    }
    output, _ = mtp_recurrent._mtp_fused_recurrent_gated_delta_rule_autotuned(**kwargs)
    return output, state, kwargs


_RETAINED_CONFIGS = mtp_recurrent._get_mtp_fused_recurrent_configs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fixed_seq_len", [0, 4])
@pytest.mark.parametrize("run_config", _RETAINED_CONFIGS)
def test_retained_configs_match_fallback(run_config, fixed_seq_len):
    case = _make_fixed_case()
    fallback = mtp_recurrent._default_mtp_fused_recurrent_config(128)
    expected_output, expected_state, _ = _launch(case, fallback, fixed_seq_len=fixed_seq_len)
    output, state, _ = _launch(case, run_config, fixed_seq_len=fixed_seq_len)

    torch.testing.assert_close(output.float(), expected_output.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state.float(), expected_state.float(), atol=5.0, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tuner_warmup_does_not_mutate_caller_state():
    case = _make_fixed_case()
    fallback = mtp_recurrent._default_mtp_fused_recurrent_config(128)
    _, state, kwargs = _launch(case, fallback, fixed_seq_len=4)
    state.copy_(case[3])
    state_before = state.clone()
    tuner = mtp_recurrent._mtp_fused_recurrent_gated_delta_rule_autotuned

    tuner.kernel_warmup(tuner._static_key(**kwargs), **kwargs)

    torch.testing.assert_close(state, state_before, atol=0, rtol=0)
