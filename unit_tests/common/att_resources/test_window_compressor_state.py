"""统一窗口状态：K-pool 的组边界及不同 MTP 容量间的 PD 恢复。"""

from dataclasses import replace
from functools import partial
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.attention_state.bounded_att import WindowCompressorState, WindowCompressorStateConfig
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import FullAttKVAllocator
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.server.core.objs.start_args_type import StartArgs


def make_window(
    device: str = "cpu",
    mtp_step: int = 2,
    dtype: torch.dtype = torch.float32,
) -> WindowCompressorState:
    return WindowCompressorState(
        config=WindowCompressorStateConfig(
            logical_layer_ids=(3, 7),
            window_size=4,
            compress_stride=4,
            candidate_dim=2,
            target="indexer",
            dtype=dtype,
        ),
        request_capacity=3,
        mtp_step=mtp_step,
        device=device,
    )


def window_layout(state: WindowCompressorState) -> StateTransferLayout:
    regions = tuple(
        replace(region, byte_start=region.byte_start + 16, byte_end=region.byte_end + 16)
        for region in state.get_pd_transfer_regions()
    )
    return StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=None,
        byte_start=16,
        byte_end=regions[-1].byte_end,
        regions=regions,
    )


def test_nonoverlapping_window_needs_no_cpu_snapshot() -> None:
    state = make_window()
    assert state.buffer.shape == (2, 3, 6, 4)
    state.buffer.fill_(17)
    state.reset_bounded_att_state([0, 1])
    assert not state.buffer[:, :2, :, :2].any()
    assert torch.isneginf(state.buffer[:, :2, :, 2:]).all()
    assert (state.buffer[:, 2] == 17).all()
    state.initialize_page_pools(big_page_capacity=5, small_page_capacity=2)
    assert state.big_pages_pool is state.small_pages_pool is None
    assert state.get_cpu_transfer_regions() == ()
    # 组边界不依赖此前窗口；下一组四个输入均由新数据覆盖，无需读取快照。
    state.restore_bounded_att_runtime_from_pages(
        req_indices=[2],
        state_token_lens=[8],
        page_kind="big",
        page_slot_ids=[0],
    )
    with pytest.raises(AssertionError, match="stride"):
        state.restore_bounded_att_runtime_from_pages(
            req_indices=[2],
            state_token_lens=[7],
            page_kind="big",
            page_slot_ids=[0],
        )


@pytest.mark.parametrize("window_size,compress_stride", [(4, 4), (8, 4), (128, 128)])
@pytest.mark.parametrize("page_kind", ["small", "cpu"])
def test_window_cache_boundaries_must_close_compressor_groups(
    window_size: int,
    compress_stride: int,
    page_kind: str,
    cpu_args: StartArgs,
) -> None:
    cpu_args.linear_att_hash_page_size = compress_stride
    cpu_args.cpu_cache_token_page_size = 2 * compress_stride
    if page_kind == "small":
        cpu_args.linear_att_hash_page_size += 1
    else:
        cpu_args.cpu_cache_token_page_size += 1
    state = WindowCompressorState(
        config=replace(make_window().config, window_size=window_size, compress_stride=compress_stride),
        request_capacity=3,
        device="meta",
    )
    # 即使不占 CPU 页，也不能在未完成的压缩组内恢复，否则缺少该组历史。
    with pytest.raises(AssertionError, match="pages must align to the compressor stride"):
        state.get_cpu_transfer_regions()


@pytest.mark.parametrize("window_size,compress_stride", [(4, 4), (8, 4), (128, 128)])
def test_manager_allocates_page_tensors_only_for_snapshot_states(
    window_size: int,
    compress_stride: int,
    cpu_args: StartArgs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_args.linear_att_hash_page_size = 128
    cpu_args.linear_att_page_block_num = 2
    cpu_args.cpu_cache_token_page_size = 256
    state = WindowCompressorState(
        config=replace(make_window().config, window_size=window_size, compress_stride=compress_stride),
        request_capacity=3,
        device="cpu",
    )
    initialize_pools = Mock(wraps=state.initialize_page_pools)
    monkeypatch.setattr(state, "initialize_page_pools", initialize_pools)
    monkeypatch.setattr(
        "lightllm.common.attention_state.manager.FullAttKVAllocator",
        partial(FullAttKVAllocator, pin_memory=False),
    )
    manager = AttentionStateManager(request_capacity=3)
    manager.register_state("main", GQAFullAttentionState(
        layout_config=GQAAttentionConfig(
            logical_layer_groups=((3,),),
            global_head_num=1,
            head_dim=4,
            dtype=torch.bfloat16,
        ),
        device="meta",
    ))
    manager.register_state("main", state)
    manager.finish_state_registration()
    manager.allocate_storage(
        token_capacity=512,
        page_size=4,
        big_page_tokens=256,
        small_page_capacity=3,
        device="cpu",
    )
    if window_size == compress_stride:
        initialize_pools.assert_not_called()
        assert state.big_pages_pool is state.small_pages_pool is None
        assert state.state_name not in manager.get_cpu_page_layouts()
    else:
        initialize_pools.assert_called_once()
        assert state.big_pages_pool.buffer.shape == (4, 2, window_size - compress_stride, 4)
        assert state.small_pages_pool.buffer.shape == (3, 2, window_size - compress_stride, 4)
        assert state.state_name in manager.get_cpu_page_layouts()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("source_step,target_step", [(0, 0), (0, 5), (2, 0), (1, 3)])
@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 11, 12])
def test_pd_window_restores_different_mtp_rings(
    dtype: torch.dtype,
    source_step: int,
    target_step: int,
    length: int,
) -> None:
    source = make_window(device="cuda", mtp_step=source_step, dtype=dtype)
    target = make_window(device="cuda", mtp_step=target_step, dtype=dtype)
    source.buffer.copy_(torch.arange(source.buffer.numel(), device="cuda").reshape_as(source.buffer))
    target.buffer.fill_(37)
    layout = window_layout(source)
    assert layout == window_layout(target)
    pages = torch.full((3, layout.byte_end + 16), 253, dtype=torch.uint8, device="cuda")
    source.write_to_pd_page(
        req_idx=1,
        pd_page_buffers=pages,
        pd_page_index=1,
        state_token_len=length,
        layout=layout,
    )
    target.read_from_pd_page(
        pd_page_buffers=pages,
        req_idx=2,
        pd_page_index=1,
        state_token_len=length,
        layout=layout,
    )
    expected = torch.full_like(target.buffer[:, 2], 37)
    for position in range(length - 4, length):
        slot = position % (4 + target_step)
        expected[:, slot] = 0 if position < 0 else source.buffer[:, 1, position % (4 + source_step)]
    torch.testing.assert_close(target.buffer[:, 2], expected, rtol=0, atol=0)
    assert (target.buffer[:, :2] == 37).all()
    assert (pages[0] == 253).all() and (pages[2] == 253).all()
    assert (pages[1, :16] == 253).all() and (pages[1, layout.byte_end:] == 253).all()
