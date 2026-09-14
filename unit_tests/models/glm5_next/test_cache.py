import dataclasses
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import Glm5NextMemManager
from lightllm.common.req_manager import Glm5NextReqManager
from lightllm.common.state_cache_manager import Glm5NextCacheConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_nope_cache_config_uses_native_mla_width():
    config = Glm5NextCacheConfig.from_model_config(
        {
            "kv_lora_rank": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            "index_kpool": 4,
            "index_head_dim": 128,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [0, 1, 2],
            },
        },
        StartArgs(tp=4, data_type="bfloat16"),
    )
    assert config.full_att_head_dim == 584
    assert config.full_att_head_dim * config.full_att_dtype.itemsize == 512 * 2 + 144


@pytest.mark.parametrize("small_page", [False, True])
@pytest.mark.parametrize("tp_world_size", [1, 4])
def test_hybrid_checkpoint_restore_and_packed_kv_copy(monkeypatch, small_page, tp_world_size):
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    args = StartArgs(
        tp=tp_world_size,
        data_type="bfloat16",
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        cpu_cache_token_page_size=8,
    )
    set_unique_server_name(args)
    get_env_start_args.cache_clear()
    set_env_start_args(dataclasses.asdict(args))
    config = Glm5NextCacheConfig(
        tp_world_size=tp_world_size,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=584,
        global_linear_k_heads=2 * tp_world_size,
        global_linear_v_heads=2 * tp_world_size,
        num_linear_k_heads=2,
        num_linear_v_heads=2,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        linear_layer_num=3,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4,
        all_layer_num=4,
    )
    mem = Glm5NextMemManager(16, torch.bfloat16, 1, 584, 1, config)
    req = Glm5NextReqManager(3, 16, mem, config)
    att_kv = mem.get_att_input_params(3)
    assert att_kv.shape == (17, 1, 512)
    assert att_kv.stride(0) == 584
    index_bytes = mem.get_indexer_k_buffer(3)
    index_bytes.random_(0, 256)
    expected_index_bytes = index_bytes.clone()
    new_kv = torch.randn(2, 1, 512, dtype=torch.bfloat16, device="cuda")
    destinations = torch.tensor([5, 9], dtype=torch.int32, device="cuda")
    mem.operator.copy_kv_to_mem_manager(3, destinations, new_kv)
    assert torch.equal(att_kv[destinations], new_kv)
    assert torch.equal(index_bytes, expected_index_bytes)
    assert req.get_indexer_tail_buffer(3).shape == (4, 4, 256)
    cache = req.create_small_page_cache_manager(2) if small_page else mem.big_page_buffers
    slot = cache.alloc_one_state_cache()
    source_req = SimpleNamespace(req_idx=0)
    req.init_hybrid_attention_state(source_req)
    req.req_to_conv_state.buffer[:, 0].normal_()
    req.req_to_ssm_state.buffer[:, 0].normal_()
    conv = req.req_to_conv_state.buffer[:, 0].clone()
    ssm = req.req_to_ssm_state.buffer[:, 0].clone()
    if small_page:
        req.save_state(0, slot, cache)
    else:
        req.save_big_page_states(torch.tensor([0], dtype=torch.int32, device="cuda"), [0], [slot])
    torch.cuda.synchronize()
    req.req_to_conv_state.buffer[:, 0].zero_()
    req.req_to_ssm_state.buffer[:, 0].zero_()
    req.req_to_indexer_tail.buffer.normal_()
    unchanged = req.req_to_indexer_tail.buffer[:, [0, 1, 3]].clone()
    dest_req = SimpleNamespace(req_idx=2, shared_kv_node=SimpleNamespace(small_page_buffer_idx=slot))
    if small_page:
        req.restore_small_page_state(dest_req)
    else:
        req.restore_big_page_state(slot, dest_req)
    torch.cuda.synchronize()
    assert torch.equal(req.req_to_conv_state.buffer[:, 2], conv)
    assert torch.equal(req.req_to_ssm_state.buffer[:, 2], ssm)
    # Both page sizes are aligned to complete pools. Restoring a prefix must
    # discard stale tail values from a previously allocated request slot.
    assert not req.req_to_indexer_tail.buffer[:, 2].any()
    assert torch.equal(req.req_to_indexer_tail.buffer[:, [0, 1, 3]], unchanged)
    req.req_to_indexer_tail.buffer[:, 2].normal_()
    req.init_hybrid_attention_state(dest_req)
    assert not req.req_to_conv_state.buffer[:, 2].any()
    assert not req.req_to_ssm_state.buffer[:, 2].any()
    assert not req.req_to_indexer_tail.buffer[:, 2].any()
    assert torch.equal(req.req_to_indexer_tail.buffer[:, [0, 1, 3]], unchanged)
    # KV moves carry MLA latents and pooled FP8 bytes; raw keys/gates only
    # belong to live requests. Bytewise equality checks FP8 scale integrity.
    packed_bytes = mem.kv_buffer.view(torch.uint8)
    packed_bytes[:, 0].random_(0, 256)
    mem.operator.copy_mem_to_mem(torch.tensor([0]), torch.tensor([7]))
    assert torch.equal(packed_bytes[:, 0], packed_bytes[:, 7])
    assert mem.get_cell_size() == 584 * 2
    assert config.get_cpu_cache_full_att_bytes() == mem.get_cell_size() * 8

    from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
        copy_kv_buffer_to_cpu_cache,
        copy_cpu_cache_to_kv_buffer,
    )

    cpu_pages = torch.zeros((1, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)
    indexes = torch.arange(8, dtype=torch.int32, device="cuda")
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    big = mem.big_page_buffers
    packed_bytes[:, :8].random_(0, 256)
    big.conv_state_cache.buffer[0].normal_()
    big.ssm_state_cache.buffer[0].normal_()
    expected_kv = packed_bytes[:, :8].clone()
    expected_conv = big.conv_state_cache.buffer[0].clone()
    expected_ssm = big.ssm_state_cache.buffer[0].clone()
    common = dict(
        mem_indexes=indexes,
        page_indexes=zero,
        big_page_buffer_ids=zero,
        cpu_kv_conv_state=big.conv_state_cache.buffer,
        cpu_kv_ssm_state=big.ssm_state_cache.buffer,
        cpu_cache_tensor=cpu_pages,
        tp_world_size=tp_world_size,
        big_page_token_num=8,
        linear_config=config,
    )
    # Simulate TP writers to a shared CPU page: MLA/index KV is replicated,
    # while each rank must retain its own KDA checkpoint region.
    expected_states = []
    for rank in range(tp_world_size):
        big.conv_state_cache.buffer[0].copy_(expected_conv + rank)
        big.ssm_state_cache.buffer[0].copy_(expected_ssm + rank)
        expected_states.append((big.conv_state_cache.buffer[0].clone(), big.ssm_state_cache.buffer[0].clone()))
        copy_kv_buffer_to_cpu_cache(page_readies=ready, gpu_kv_full_att_state=mem.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
    for rank, (conv, ssm) in enumerate(expected_states):
        packed_bytes[:, :8].zero_()
        big.conv_state_cache.buffer[0].zero_()
        big.ssm_state_cache.buffer[0].zero_()
        copy_cpu_cache_to_kv_buffer(gpu_full_att_kv_state=mem.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
        assert torch.equal(packed_bytes[:, :8], expected_kv)
        assert torch.equal(big.conv_state_cache.buffer[0], conv)
        assert torch.equal(big.ssm_state_cache.buffer[0], ssm)


@pytest.mark.parametrize("gate_type", ["silu", "sigmoid"])
def test_gated_norm_gate_type_and_strided_gate(gate_type):
    from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward

    x = torch.randn(12, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 8, 128, device="cuda", dtype=torch.bfloat16)[:, :4]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    actual = gated_rmsnorm_forward(x, weight, None, 1e-5, gate, gate_type=gate_type)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * weight.float()
    z = gate.reshape(12, 128).float()
    expected *= z.sigmoid() if gate_type == "sigmoid" else torch.nn.functional.silu(z)
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0.008, rtol=0.008)


@pytest.mark.parametrize("add_one", [False, True])
def test_clamped_swiglu_preserves_gpt_oss_default(add_one):
    from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

    x = torch.linspace(-25, 25, 2048, device="cuda", dtype=torch.bfloat16).view(4, 512)
    out = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    kwargs = {} if add_one else {"clamp_up_add_one": False}
    alpha, limit = (1.702, 7.0) if add_one else (1.0, 10.0)
    silu_and_mul_fwd(x, out, limit=limit, alpha=alpha, **kwargs)
    gate, up = x.float().chunk(2, -1)
    gate = gate.clamp(max=limit)
    gate = (gate * torch.sigmoid(alpha * gate)).bfloat16().float()
    expected = gate * (up.clamp(-limit, limit) + int(add_one))
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.008, rtol=0.008)
