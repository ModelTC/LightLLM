import json
import os

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
    copy_cpu_cache_to_kv_buffer,
    copy_kv_buffer_to_cpu_cache,
)
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.models.kimi_linear.triton_kernel.fla.ops.kda import (
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)
from lightllm.utils.config_utils import is_linear_att_mixed_model
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture
def linear_cache_args(monkeypatch):
    args = {
        "cpu_cache_token_page_size": 4,
        "linear_att_hash_page_size": 2,
        "linear_att_page_block_num": 2,
        "data_type": "bfloat16",
        "linear_att_ssm_data_type": "float32",
        "model_dir": "unused",
        "tp": 2,
        "dp": 1,
        "mtp_mode": None,
        "mtp_step": 0,
    }
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
    get_env_start_args.cache_clear()
    yield args
    get_env_start_args.cache_clear()


def test_kimi_linear_uses_lightllm_hybrid_prefix_cache(tmp_path, linear_cache_args):
    config = {
        "model_type": "kimi_linear",
        "num_hidden_layers": 5,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "linear_attn_config": {
            "full_attn_layers": [2, 4],
            "kda_layers": [1, 3, 5],
            "num_heads": 4,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    linear_cache_args["model_dir"] = str(tmp_path)
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(linear_cache_args)
    get_env_start_args.cache_clear()
    is_linear_att_mixed_model.cache_clear()

    assert is_linear_att_mixed_model(str(tmp_path))
    cache_config = LinearAttCacheConfig.load_from_args()
    assert cache_config.full_attention_layers == (1, 3)
    assert cache_config.linear_layer_num == 3
    assert cache_config.full_att_kv_factor == 1
    assert [cache_config.get_full_attention_layer_index(i) for i in (1, 3)] == [0, 1]
    assert [cache_config.get_linear_attention_layer_index(i) for i in (0, 2, 4)] == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_compressed_mla_and_kda_state_round_trip_through_lightllm_cpu_cache(linear_cache_args):
    cache_config = LinearAttCacheConfig(
        tp_world_size=2,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=8,
        global_linear_k_heads=4,
        global_linear_v_heads=4,
        num_linear_k_heads=2,
        num_linear_v_heads=2,
        head_linear_k_dim=4,
        head_linear_v_dim=4,
        conv_kernel_size=3,
        linear_layer_num=3,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=1,
        all_layer_num=5,
        full_attention_layers=(1, 3),
        full_att_kv_factor=1,
    )

    token_capacity = 8
    full_state = torch.arange(2 * (token_capacity + 1) * 8, device="cuda", dtype=torch.bfloat16).view(
        2, token_capacity + 1, 1, 8
    )
    conv_state = torch.arange(2 * 3 * 24 * 2, dtype=torch.bfloat16).view(2, 3, 24, 2).pin_memory()
    ssm_state = torch.arange(2 * 3 * 2 * 4 * 4, dtype=torch.float32).view(2, 3, 2, 4, 4).pin_memory()
    expected_full = full_state.clone()
    expected_conv = conv_state[0].clone()
    expected_ssm = ssm_state[0].clone()

    mem_indexes = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int64)
    page_indexes = torch.tensor([0], device="cuda", dtype=torch.int32)
    page_readies = torch.zeros(1, device="cuda", dtype=torch.int32)
    big_page_buffer_ids = torch.tensor([0], device="cuda", dtype=torch.int64)
    cpu_cache = torch.empty((1, cache_config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)

    copy_kv_buffer_to_cpu_cache(
        mem_indexes=mem_indexes,
        page_indexes=page_indexes,
        page_readies=page_readies,
        big_page_buffer_ids=big_page_buffer_ids,
        gpu_kv_full_att_state=full_state,
        cpu_kv_conv_state=conv_state,
        cpu_kv_ssm_state=ssm_state,
        cpu_cache_tensor=cpu_cache,
        tp_rank=0,
        tp_world_size=2,
        big_page_token_num=4,
        linear_config=cache_config,
    )
    torch.cuda.synchronize()

    full_state[:, mem_indexes].zero_()
    conv_state[0].zero_()
    ssm_state[0].zero_()
    copy_cpu_cache_to_kv_buffer(
        mem_indexes=mem_indexes,
        big_page_buffer_ids=big_page_buffer_ids,
        page_indexes=page_indexes,
        gpu_full_att_kv_state=full_state,
        cpu_kv_conv_state=conv_state,
        cpu_kv_ssm_state=ssm_state,
        cpu_cache_tensor=cpu_cache,
        tp_rank=0,
        tp_world_size=2,
        big_page_token_num=4,
        linear_config=cache_config,
    )
    torch.cuda.synchronize()

    assert torch.equal(full_state[:, mem_indexes], expected_full[:, mem_indexes])
    assert torch.equal(conv_state[0], expected_conv)
    assert torch.equal(ssm_state[0], expected_ssm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_kda_decode_accepts_zero_based_lightllm_request_slot():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    heads = key_dim = value_dim = 4
    state = torch.randn(2, heads, value_dim, key_dim, device="cuda", dtype=torch.float32)
    state[1].copy_(state[0])
    initial_state = state.clone()

    q = torch.randn(1, 1, heads, key_dim, device="cuda", dtype=dtype).repeat(2, 1, 1, 1)
    k = torch.randn(1, 1, heads, key_dim, device="cuda", dtype=dtype).repeat(2, 1, 1, 1)
    v = torch.randn(1, 1, heads, value_dim, device="cuda", dtype=dtype).repeat(2, 1, 1, 1)
    g = -torch.rand(1, 1, heads, key_dim, device="cuda", dtype=torch.float32).repeat(2, 1, 1, 1)
    beta = torch.rand(1, 1, heads, device="cuda", dtype=torch.float32).repeat(2, 1, 1)

    out, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        inplace_final_state=True,
        ssm_state_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(out[0], out[1])
    assert torch.equal(state[0], state[1])
    assert torch.count_nonzero(out[0]) > 0
    assert not torch.equal(state[0], initial_state[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_kimi_packed_kda_decode_matches_contiguous_reference(batch):
    torch.manual_seed(2)
    dtype = torch.bfloat16
    heads, head_dim = 8, 128
    slots = batch + 3
    mixed_qkv = torch.randn(batch, 3 * heads * head_dim, device="cuda", dtype=dtype)
    q, k, v = mixed_qkv.split([heads * head_dim] * 3, dim=-1)
    q = q.view(batch, 1, heads, head_dim)
    k = k.view(batch, 1, heads, head_dim)
    v = v.view(batch, 1, heads, head_dim)
    gate = -torch.rand(batch, 1, heads, head_dim, device="cuda", dtype=torch.float32)
    beta = torch.rand(batch, 1, heads, device="cuda", dtype=torch.float32)
    indices = torch.randperm(slots, device="cuda")[:batch].to(torch.int64)
    initial_state = torch.randn(slots, heads, head_dim, head_dim, device="cuda", dtype=torch.float32)

    reference_state = initial_state.clone()
    reference, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=reference_state,
        inplace_final_state=True,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    packed_state = initial_state.clone()
    packed = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        g=gate,
        beta=beta,
        initial_state=packed_state,
        ssm_state_indices=indices,
        head_dim=head_dim,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(reference, packed)
    assert torch.equal(reference_state, packed_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_sigmoid_gated_rmsnorm_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    actual = gated_rmsnorm_forward(x, weight, bias=None, eps=eps, z=gate, activation="sigmoid")
    normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    expected = normalized * weight.float() * gate.float().sigmoid()

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
