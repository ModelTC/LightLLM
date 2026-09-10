import dataclasses
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.models.glm5_next.cache_config import Glm5NextCacheConfig
from lightllm.models.glm5_next.mem_manager import Glm5NextMemManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args, set_env_start_args, set_unique_server_name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
        full_att_head_dim=904,
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
    monkeypatch.setattr("lightllm.common.state_cache_manager.LinearAttCacheConfig.load_from_args", lambda: config)
    mem = Glm5NextMemManager(16, torch.bfloat16, 1, 904, 1, config)
    req = ReqManagerForMamba(3, 16, mem, config)
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
    dest_req = SimpleNamespace(req_idx=2, shared_kv_node=SimpleNamespace(small_page_buffer_idx=slot))
    if small_page:
        req.restore_small_page_state(dest_req)
    else:
        req.restore_big_page_state(slot, dest_req)
    torch.cuda.synchronize()
    assert torch.equal(req.req_to_conv_state.buffer[:, 2], conv)
    assert torch.equal(req.req_to_ssm_state.buffer[:, 2], ssm)
    req.init_hybrid_attention_state(dest_req)
    assert not req.req_to_conv_state.buffer[:, 2].any()
    assert not req.req_to_ssm_state.buffer[:, 2].any()
    # KV moves must carry raw index keys, compression gates and pooled FP8
    # bytes together; bytewise equality catches omissions and scale corruption.
    packed_bytes = mem.kv_buffer.view(torch.uint8)
    packed_bytes[:, 0].random_(0, 256)
    mem.operator.copy_mem_to_mem(torch.tensor([0]), torch.tensor([7]))
    assert torch.equal(packed_bytes[:, 0], packed_bytes[:, 7])
    assert mem.get_cell_size() == 904 * 2
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


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_gated_norm_activation_and_strided_gate(activation):
    from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward

    x = torch.randn(12, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 8, 128, device="cuda", dtype=torch.bfloat16)[:, :4]
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    actual = gated_rmsnorm_forward(x, weight, None, 1e-5, gate, activation=activation)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * weight.float()
    z = gate.reshape(12, 128).float()
    expected *= z.sigmoid() if activation == "sigmoid" else torch.nn.functional.silu(z)
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0.008, rtol=0.008)


@pytest.mark.parametrize("add_one", [False, True])
def test_clamped_swiglu_preserves_gpt_oss_default(add_one):
    from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

    x = torch.linspace(-25, 25, 2048, device="cuda", dtype=torch.bfloat16).view(4, 512)
    out = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    kwargs = {} if add_one else {"clamp_up_add_one": False}
    silu_and_mul_fwd(x, out, limit=10.0, alpha=1.0, **kwargs)
    gate, up = x.float().chunk(2, -1)
    gate = torch.nn.functional.silu(gate.clamp(max=10)).bfloat16().float()
    expected = gate * (up.clamp(-10, 10) + int(add_one))
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.008, rtol=0.008)
