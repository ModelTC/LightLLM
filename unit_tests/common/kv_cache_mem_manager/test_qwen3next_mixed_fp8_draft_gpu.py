from types import SimpleNamespace

import pytest
import torch

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextLinearAttPageHelper,
    Qwen3NextMemManager,
    _FP8StaticPerHeadQuantLinearAttMemOperator,
)
import lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager as qwen_mem
import lightllm.common.linear_att_cache_manager.config_objs as config_objs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _config(tp):
    return LinearAttCacheConfig(
        tp_world_size=tp,
        full_att_all_num_kv_heads=4,
        full_att_dtype=torch.uint8,
        full_att_num_kv_heads=4 // tp,
        full_att_head_dim=2,
        global_linear_k_heads=4,
        global_linear_v_heads=4,
        num_linear_k_heads=4 // tp,
        num_linear_v_heads=4 // tp,
        head_linear_k_dim=2,
        head_linear_v_dim=2,
        conv_kernel_size=2,
        linear_layer_num=1,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.bfloat16,
        full_attention_interval=2,
        all_layer_num=2,
        draft_full_att_kv_layer_num=1,
        draft_full_att_dtype=torch.bfloat16,
    )


def _manager(config, size=8, req_slots=4):
    manager = object.__new__(Qwen3NextMemManager)
    manager.linear_config = config
    manager.target_full_att_layer_num = config.get_target_full_att_kv_layer_num()
    manager.head_dim = config.full_att_head_dim
    manager.dtype = torch.uint8
    manager.size = size
    manager.HOLD_TOKEN_MEMINDEX = size
    manager.kv_buffer = torch.zeros(
        (manager.target_full_att_layer_num, size + 1, 2 * config.full_att_num_kv_heads, manager.head_dim),
        device="cuda",
        dtype=torch.uint8,
    )
    manager.draft_kv_buffer = torch.zeros(
        (config.draft_full_att_kv_layer_num, size + 1, 2 * config.full_att_num_kv_heads, manager.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_dim = config.get_conv_dim()
    manager.req_to_conv_state = SimpleNamespace(
        buffer=torch.zeros((1, req_slots, conv_dim, 3), device="cuda", dtype=torch.bfloat16)
    )
    manager.req_to_ssm_state = SimpleNamespace(
        buffer=torch.zeros(
            (1, req_slots * 3, config.num_linear_v_heads, config.head_linear_k_dim, config.head_linear_v_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    return manager


def _global_kv(mems, attr, indexes):
    pieces = [getattr(mem, attr)[:, indexes] for mem in mems]
    # Local buffers are [K-local-heads, V-local-heads].  Rebuild the global
    # page order as all K heads followed by all V heads across TP ranks.
    local_heads = pieces[0].shape[2] // 2
    keys = torch.cat([piece[:, :, :local_heads] for piece in pieces], dim=2)
    values = torch.cat([piece[:, :, local_heads:] for piece in pieces], dim=2)
    return torch.cat([keys, values], dim=2)


def test_mixed_fp8_bf16_page_io_tp2_to_tp4_and_linear_state(monkeypatch):
    # This is a TP-layout test on one GPU: two source ranks write a global page;
    # four destination-rank slices read it back from the same physical page.
    monkeypatch.setattr(qwen_mem, "get_env_start_args", lambda: SimpleNamespace(mtp_step=1))
    src = [_manager(_config(2)), _manager(_config(2))]
    dst = [_manager(_config(4)) for _ in range(4)]
    src_indexes = [1, 4, 6]
    dst_indexes = [0, 3, 5]
    for rank, mem in enumerate(src):
        mem.kv_buffer.copy_(torch.arange(mem.kv_buffer.numel(), device="cuda", dtype=torch.uint8).view_as(mem.kv_buffer) + rank)
        mem.draft_kv_buffer.copy_(
            (torch.arange(mem.draft_kv_buffer.numel(), device="cuda").view_as(mem.draft_kv_buffer) + rank * 100).to(torch.bfloat16)
        )
        mem.req_to_conv_state.buffer.copy_(
            (torch.arange(mem.req_to_conv_state.buffer.numel(), device="cuda").view_as(mem.req_to_conv_state.buffer) + rank * 1000).to(torch.bfloat16)
        )
        mem.req_to_ssm_state.buffer.copy_(
            (torch.arange(mem.req_to_ssm_state.buffer.numel(), device="cuda").view_as(mem.req_to_ssm_state.buffer) + rank * 1000).to(torch.bfloat16)
        )
    hold_target = src[0].kv_buffer[:, src[0].HOLD_TOKEN_MEMINDEX].clone()
    hold_draft = src[0].draft_kv_buffer[:, src[0].HOLD_TOKEN_MEMINDEX].clone()
    expected_target = _global_kv(src, "kv_buffer", src_indexes).clone()
    expected_draft = _global_kv(src, "draft_kv_buffer", src_indexes).clone()

    state_sized_page = _manager(_config(2))
    state_sized_page.alloc_paged_kv_move_buffer(page_num=1, page_size=1)
    assert state_sized_page.kv_move_buffer.shape[-1] >= Qwen3NextLinearAttPageHelper(state_sized_page).state_nbytes
    assert state_sized_page.kv_move_buffer.shape[-1] > (
        state_sized_page._mixed_target_page_bytes + state_sized_page._mixed_draft_page_bytes
    )

    src[0].alloc_paged_kv_move_buffer(page_num=2, page_size=4)
    src[0].write_mem_to_page_kv_move_buffer(src_indexes, 0, 0, src, 2)
    src[0].write_mem_to_page_kv_move_buffer([], 1, 0, src, 2, page_kind="linear_att_state", req_idx=1)

    dst[0].alloc_paged_kv_move_buffer(page_num=3, page_size=4)
    dst[0].kv_move_buffer[:2].copy_(src[0].kv_move_buffer)
    for mem in dst:
        mem.kv_buffer.zero_()
        mem.draft_kv_buffer.zero_()
    dst[0].read_page_kv_move_buffer_to_mem(dst_indexes, 0, 0, dst, 4)
    # Destination MTP step is two, so page helper uses a different SSM stride.
    monkeypatch.setattr(qwen_mem, "get_env_start_args", lambda: SimpleNamespace(mtp_step=2))
    dst[0].read_page_kv_move_buffer_to_mem([], 1, 0, dst, 4, page_kind="linear_att_state", req_idx=2)
    dst[0].write_mem_to_page_kv_move_buffer([], 2, 0, dst, 4, page_kind="linear_att_state", req_idx=2)

    torch.cuda.synchronize()
    assert torch.equal(_global_kv(dst, "kv_buffer", dst_indexes), expected_target)
    assert torch.equal(_global_kv(dst, "draft_kv_buffer", dst_indexes), expected_draft)
    assert not torch.any(_global_kv(dst, "kv_buffer", [1]))
    assert not torch.any(_global_kv(dst, "draft_kv_buffer", [1]))
    assert torch.equal(src[0].kv_buffer[:, src[0].HOLD_TOKEN_MEMINDEX], hold_target)
    assert torch.equal(src[0].draft_kv_buffer[:, src[0].HOLD_TOKEN_MEMINDEX], hold_draft)
    # The page is sized for the larger of KV and global linear state, and the
    # destination receives the source's global state partition on each rank.
    assert src[0].kv_move_buffer.shape[-1] >= src[0]._mixed_target_page_bytes + src[0]._mixed_draft_page_bytes
    src_conv_page, src_ssm_page = Qwen3NextLinearAttPageHelper(dst[0]).view_page_to_linear_att_state(1)
    dst_conv_page, dst_ssm_page = Qwen3NextLinearAttPageHelper(dst[0]).view_page_to_linear_att_state(2)
    assert torch.equal(dst_conv_page, src_conv_page)
    assert torch.equal(dst_ssm_page, src_ssm_page)


def test_pinned_cpu_cache_two_segments_and_dual_buffer_move(monkeypatch):
    # Actual CPU-cache Triton wrappers with a small pinned page.  The final
    # -1 token is a tail-page hole and must not overwrite any GPU slot.
    args = SimpleNamespace(linear_att_page_block_num=4, linear_att_hash_page_size=1, cpu_cache_token_page_size=4)
    monkeypatch.setattr(config_objs, "get_env_start_args", lambda: args)
    config = _config(2)
    target = torch.arange(1 * 9 * 4 * 2, device="cuda", dtype=torch.uint8).view(1, 9, 4, 2)
    draft = torch.arange(1 * 9 * 4 * 2, device="cuda").view(1, 9, 4, 2).to(torch.bfloat16)
    conv = torch.arange(12, dtype=torch.bfloat16).view(1, 1, 12, 1).pin_memory()
    ssm = torch.arange(8, dtype=torch.bfloat16).view(1, 1, 2, 2, 2).pin_memory()
    cpu_cache = torch.zeros((1, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8).pin_memory()
    mem_indexes = torch.tensor([1, 4, 6, -1], device="cuda", dtype=torch.int32)
    page_indexes = torch.tensor([0], device="cuda", dtype=torch.int64)
    page_readies = torch.tensor([0], device="cuda", dtype=torch.int32)
    page_ids = torch.tensor([0], device="cuda", dtype=torch.int64)
    expected_target = target[:, [1, 4, 6]].clone()
    expected_draft = draft[:, [1, 4, 6]].clone()
    expected_conv, expected_ssm = conv.clone(), ssm.clone()

    from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
        copy_cpu_cache_to_kv_buffer,
        copy_kv_buffer_to_cpu_cache,
    )

    skipped_cache = torch.zeros(cpu_cache.shape, dtype=torch.uint8).pin_memory()
    skipped_readies = torch.tensor([1], device="cuda", dtype=torch.int32)
    copy_kv_buffer_to_cpu_cache(
        mem_indexes, page_indexes, skipped_readies, page_ids, target, conv, ssm, skipped_cache, 0, 2, 4, config,
        full_att_bytes=config.get_cpu_cache_target_full_att_bytes(),
    )
    torch.cuda.synchronize()
    assert not torch.any(skipped_cache)

    copy_kv_buffer_to_cpu_cache(
        mem_indexes, page_indexes, page_readies, page_ids, target, conv, ssm, cpu_cache, 0, 2, 4, config,
        full_att_bytes=config.get_cpu_cache_target_full_att_bytes(),
    )
    copy_kv_buffer_to_cpu_cache(
        mem_indexes, page_indexes, page_readies, page_ids, draft, conv, ssm, cpu_cache, 0, 2, 4, config,
        full_att_byte_offset=config.get_cpu_cache_target_full_att_bytes(),
        full_att_bytes=config.get_cpu_cache_draft_full_att_bytes(),
        copy_linear_att_state=False,
    )
    target.zero_()
    draft.zero_()
    conv.zero_()
    ssm.zero_()
    copy_cpu_cache_to_kv_buffer(
        mem_indexes, page_ids, page_indexes, target, conv, ssm, cpu_cache, 0, 2, 4, config,
        full_att_bytes=config.get_cpu_cache_target_full_att_bytes(),
    )
    copy_cpu_cache_to_kv_buffer(
        mem_indexes, page_ids, page_indexes, draft, conv, ssm, cpu_cache, 0, 2, 4, config,
        full_att_byte_offset=config.get_cpu_cache_target_full_att_bytes(),
        full_att_bytes=config.get_cpu_cache_draft_full_att_bytes(),
        copy_linear_att_state=False,
    )
    torch.cuda.synchronize()
    assert torch.equal(target[:, [1, 4, 6]], expected_target)
    assert torch.equal(draft[:, [1, 4, 6]], expected_draft)
    assert torch.equal(conv, expected_conv)
    assert torch.equal(ssm, expected_ssm)
    assert not torch.any(target[:, [0, 2, 3, 5, 7, 8]])
    assert not torch.any(draft[:, [0, 2, 3, 5, 7, 8]])

    manager = SimpleNamespace(kv_buffer=target, draft_kv_buffer=draft)
    operator = object.__new__(_FP8StaticPerHeadQuantLinearAttMemOperator)
    operator.mem_manager = manager
    operator.copy_mem_to_mem(torch.tensor([1, 4]), torch.tensor([2, 3]))
    torch.cuda.synchronize()
    assert torch.equal(target[:, [2, 3]], target[:, [1, 4]])
    assert torch.equal(draft[:, [2, 3]], draft[:, [1, 4]])
