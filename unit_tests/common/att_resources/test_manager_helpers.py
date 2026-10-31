from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from typing import TYPE_CHECKING
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state.allocators import FullAttKVAllocator, StatePageAllocator
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.types import PageKind, StateTransferLayout
from unit_tests.common.att_resources.manager_fixtures import MainDraftTestResourceManager
from lightllm.common.attention_state.att_tail import KPoolAttentionTailState
from unit_tests.common.att_resources.test_full_state import full_state, layout_for
from unit_tests.common.att_resources.test_bounded_state import page_layout
from unit_tests.common.att_resources.test_tail_state import make_tail, tail_layout

if TYPE_CHECKING:
    from easydict import EasyDict


def composition(device: str | torch.device | int = "cpu", with_tail: bool = True, tail_pool_size: int = 4):
    return MainDraftTestResourceManager(device, with_tail=with_tail, tail_pool_size=tail_pool_size)


def layouts_for(manager: AttentionStateManager, kind: str = "cpu"):
    layouts = []
    if kind != "att_state":
        layouts += [layout_for(state) for state in (manager.main_gqa_fp32_state, manager.draft0_mla_fp32_state)]
    if kind != "kv":
        layouts += [page_layout(state) for state in (manager.main_kda_conv_fp32_ssm_fp32_state, manager.draft0_gdn_conv_fp32_ssm_fp32_state)]
        if kind == "att_state":
            layouts += [tail_layout(state) for state in manager.get_attention_tail_states("main")]
    offset = 0
    result = {}
    for layout in layouts:
        state_end = offset + layout.byte_end
        result[layout.state_name] = replace(
            layout,
            byte_start=offset + layout.byte_start,
            byte_end=state_end,
            tokens_per_page=None if kind == "att_state" else 4,
            regions=tuple(
                replace(r, byte_start=r.byte_start + offset, byte_end=r.byte_end + offset) for r in layout.regions
            ),
        )
        offset = state_end
    return result


def init_source(manager: AttentionStateManager):
    for i, state in enumerate((manager.main_gqa_fp32_state, manager.draft0_mla_fp32_state)):
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.copy_(
                        torch.arange(storage_buffer.numel(), device=storage_buffer.device).reshape_as(storage_buffer) + i
                    )
    for i, state in enumerate((manager.main_kda_conv_fp32_ssm_fp32_state, manager.draft0_gdn_conv_fp32_ssm_fp32_state)):
        state.conv_state[:, 0].fill_(11 + i)
        state.conv_state[:, 1].fill_(23 + i)
        state.ssm_state[:, :3].fill_(31 + i)
        state.ssm_state[:, 3:6].fill_(47 + i)
    for state in manager.get_attention_tail_states("main"):
        state.tail_buffer[:, 0].fill_(3)
        state.tail_buffer[:, 1].fill_(9)
    slots = manager.bounded_att_big_page_allocator.alloc(2)
    manager.save_bounded_att_runtime_to_pages(
        req_indices=[0, 1],
        state_token_lens=[4, 8],
        page_kind="big",
        page_slot_ids=slots,
    )
    if manager.main_kda_conv_fp32_ssm_fp32_state.conv_state.is_cuda:
        torch.cuda.synchronize()
    return slots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bounded transfers")
def test_shared_slots_save_restore_and_propagate_failure(monkeypatch: pytest.MonkeyPatch):
    manager = composition(device="cuda")
    slots = init_source(manager)
    manager.restore_bounded_att_runtime_from_pages(
        req_indices=[2],
        state_token_lens=[4],
        page_kind="big",
        page_slot_ids=[slots[0]],
    )
    for i, state in enumerate((manager.main_kda_conv_fp32_ssm_fp32_state, manager.draft0_gdn_conv_fp32_ssm_fp32_state)):
        assert (state.conv_state[:, 2, :, :2] == 11 + i).all()
        assert (state.ssm_state[:, 6] == 31 + i).all()

    def fail(*args: object, **kwargs: object):
        raise RuntimeError("simulated copy failure")

    monkeypatch.setattr(manager.draft0_gdn_conv_fp32_ssm_fp32_state, "save_bounded_att_runtime_to_pages", fail)
    with pytest.raises(RuntimeError):
        manager.save_bounded_att_runtime_to_pages(
            req_indices=[0, 1],
            state_token_lens=[4, 8],
            page_kind="big",
            page_slot_ids=slots,
        )
    assert manager.bounded_att_big_page_allocator.free_count == 1
    manager.bounded_att_big_page_allocator.free(slots)
    assert manager.bounded_att_big_page_allocator.free_count == 3


def test_disabled_domains():
    manager = AttentionStateManager(request_capacity=3)
    manager.register_state("main", full_state())
    manager.full_att_allocator = FullAttKVAllocator(12, page_size=4, pin_memory=False)
    manager.finish_state_registration()
    manager.reset_bounded_att_state([0])
    manager.reset_attention_tail_state([0])
    with pytest.raises(AssertionError, match="disabled"):
        manager.save_bounded_att_runtime_to_pages(
            req_indices=[0],
            state_token_lens=[4],
            page_kind="big",
            page_slot_ids=[0],
        )
    empty_manager = AttentionStateManager(request_capacity=3)
    empty_manager.finish_state_registration()
    with pytest.raises(AssertionError, match="no full-attention"):
        empty_manager.copy_full_att_tokens(torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long))


@pytest.mark.parametrize("operation", ["save_bounded_att_runtime_to_pages", "restore_bounded_att_runtime_from_pages"])
def test_page_operations_forward_each_request_length(operation: str, monkeypatch: pytest.MonkeyPatch):
    manager = composition(with_tail=False)
    calls = []

    def record_lengths(
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        calls.append((req_indices, state_token_lens, page_kind, page_slot_ids))

    for state in manager.get_bounded_attention_states():
        monkeypatch.setattr(state, operation, record_lengths)
    getattr(manager, operation)(
        req_indices=[0, 2],
        state_token_lens=[17, 29],
        page_kind="big",
        page_slot_ids=[0, 1],
    )
    assert calls == [([0, 2], [17, 29], "big", [0, 1])] * len(manager.get_bounded_attention_states())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for token indices")
def test_cpu_restore_forwards_absolute_length_for_partial_prefix(monkeypatch: pytest.MonkeyPatch):
    manager = composition(with_tail=False)
    layouts = layouts_for(manager)
    manager.bounded_att_cpu_load_big_page_slot = 2
    for state in manager.get_full_attention_states():
        monkeypatch.setattr(state, "read_from_cpu", lambda **kwargs: None)
    for state in manager.get_bounded_attention_states():
        monkeypatch.setattr(state, "read_state_pages_from_cpu", lambda **kwargs: None)
    restored = []

    def record_restore(
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        restored.append((req_indices, state_token_lens, page_kind, page_slot_ids))

    monkeypatch.setattr(manager, "restore_bounded_att_runtime_from_pages", record_restore)
    # 只加载绝对位置 [64, 70)，本批长度 6；恢复位置必须是 70，且末尾碎页使用 LOAD。
    manager.read_req_from_cpu(
        cpu_page_buffers=torch.empty((2, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8),
        cpu_page_indices=[0, 1],
        token_indices=torch.arange(6, device="cuda"),
        layouts=layouts,
        big_page_slot_ids=[0],
        req_idx=1,
        state_token_len=70,
    )
    assert restored == [([1], [70], "big", [2])]


@pytest.mark.parametrize("operation", ["write_to_pd_page", "read_from_pd_page"])
def test_pd_bounded_only_forwards_prefix_length(operation: str, monkeypatch: pytest.MonkeyPatch):
    manager = composition(with_tail=False)
    layouts = layouts_for(manager, kind="att_state")
    lengths = []

    def record_length(
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        lengths.append((req_idx, state_token_len))

    for state in manager.get_bounded_attention_states():
        monkeypatch.setattr(state, operation, record_length)
    getattr(manager, operation)(
        pd_page_buffers=torch.empty((1, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8),
        pd_page_index=0,
        kind="att_state",
        layouts=layouts,
        req_idx=2,
        state_token_len=29,
    )
    assert lengths == [(2, 29)] * len(manager.get_bounded_attention_states())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bounded transfers")
def test_reset_tail_clears_all_tail_states_and_preserves_linear():
    first = make_tail()
    second = KPoolAttentionTailState(
        config=replace(first.config, pool_size=3),
        request_capacity=first.request_capacity,
        mtp_step=first.mtp_step,
        device=first.tail_buffer.device,
    )
    first.tail_buffer.fill_(5)
    second.tail_buffer.fill_(7)
    tail_manager = AttentionStateManager(request_capacity=3)
    tail_manager.register_state("main", first)
    tail_manager.register_state("main", second)
    tail_manager.finish_state_registration()
    tail_manager.reset_attention_tail_state([0, 1])
    assert not first.tail_buffer[:, :2].any() and not second.tail_buffer[:, :2].any()
    assert (first.tail_buffer[:, 2] == 5).all() and (second.tail_buffer[:, 2] == 7).all()
    manager = composition(device="cuda")
    init_source(manager)
    # Preserve accepted-state cursors as well as both algorithms' runtime and
    # cached pages. A full reset here would silently corrupt a prefix restore.
    for state in (manager.main_kda_conv_fp32_ssm_fp32_state, manager.draft0_gdn_conv_fp32_ssm_fp32_state):
        state.mtp_state_index[1] = 2
    tensors = [
        *full_state_test_buffers(manager.main_gqa_fp32_state).values(),
        *full_state_test_buffers(manager.draft0_mla_fp32_state).values(),
    ]
    for state in (manager.main_kda_conv_fp32_ssm_fp32_state, manager.draft0_gdn_conv_fp32_ssm_fp32_state):
        tensors.extend((
            state.conv_state, state.ssm_state, state.mtp_state_index,
            state.big_pages_pool.conv_state, state.big_pages_pool.ssm_state,
            state.small_pages_pool.conv_state, state.small_pages_pool.ssm_state,
        ))
    preserved = [(tensor, tensor.contiguous().view(torch.uint8).clone()) for tensor in tensors]
    manager.reset_attention_tail_state([1])
    assert not getattr(manager, f"main_{manager.att_tail_type}_state").tail_buffer[:, 1].any()
    for tensor, snapshot in preserved:
        assert torch.equal(tensor.contiguous().view(torch.uint8), snapshot)


@pytest.mark.parametrize("device", ["cuda"])
def test_cpu_all_linear_pages_and_tail_reset_at_final_boundary(device: str | torch.device | int):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source, target = composition(device), composition(device)
    slots = init_source(source)
    layouts = layouts_for(source)
    pages = torch.zeros(
        (3, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8, pin_memory=device == "cuda"
    )
    source.write_req_to_cpu(
        token_indices=torch.arange(8, dtype=torch.long, device=device),
        cpu_page_buffers=pages,
        cpu_page_indices=[2, 0],
        layouts=layouts,
        big_page_slot_ids=slots,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    tail = getattr(target, f"main_{target.att_tail_type}_state")
    tail.tail_buffer.fill_(37)
    assert tail.state_name not in layouts
    target_slots = target.bounded_att_big_page_allocator.alloc(2)
    target.read_req_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=[2, 0],
        token_indices=torch.arange(
            start=4,
            end=12,
            dtype=torch.long,
            device=device,
        ),
        state_token_len=8,
        layouts=layouts,
        big_page_slot_ids=target_slots,
        req_idx=2,
    )
    for src, dst in zip(
        (source.main_kda_conv_fp32_ssm_fp32_state, source.draft0_gdn_conv_fp32_ssm_fp32_state), (target.main_kda_conv_fp32_ssm_fp32_state, target.draft0_gdn_conv_fp32_ssm_fp32_state)
    ):
        for src_slot, dst_slot in zip(slots, target_slots):
            assert torch.equal(src.big_pages_pool.conv_state[src_slot], dst.big_pages_pool.conv_state[dst_slot])
            assert torch.equal(src.big_pages_pool.ssm_state[src_slot], dst.big_pages_pool.ssm_state[dst_slot])
    assert (tail.tail_buffer[:, :2] == 37).all()
    assert not tail.tail_buffer[:, 2].any()
    for i, state in enumerate((target.main_kda_conv_fp32_ssm_fp32_state, target.draft0_gdn_conv_fp32_ssm_fp32_state)):
        assert (state.conv_state[:, 2, :, :2] == 23 + i).all()
        assert (state.ssm_state[:, 6] == 47 + i).all()
    for src, dst in zip(
        (source.main_gqa_fp32_state, source.draft0_mla_fp32_state),
        (target.main_gqa_fp32_state, target.draft0_mla_fp32_state),
    ):
        assert torch.equal(read_kv_storage(src)[:, :8], read_kv_storage(dst)[:, 4:12])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("token_count", [2, 4, 6, 8])
def test_cpu_restore_owns_partial_slot_and_restores_final_runtime(token_count: int):
    source, target = composition("cuda", with_tail=False), composition("cuda", with_tail=False)
    source_slots = init_source(source)
    layouts = layouts_for(source)
    page_count = (token_count + 3) // 4
    page_ids = [2, 0][:page_count]
    pages = torch.zeros((3, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8, pin_memory=True)
    small_slots = []
    if token_count % 4:
        small_slots = source.bounded_att_small_page_allocator.alloc(1)
        source.save_bounded_att_runtime_to_pages(
            req_indices=[page_count - 1],
            state_token_lens=[token_count],
            page_kind="small",
            page_slot_ids=small_slots,
        )
        torch.cuda.synchronize()
    source.write_req_to_cpu(
        token_indices=torch.arange(token_count, device="cuda"),
        cpu_page_buffers=pages,
        cpu_page_indices=page_ids,
        layouts=layouts,
        big_page_slot_ids=source_slots[:token_count // 4],
        small_page_slot_ids=small_slots,
    )
    torch.cuda.synchronize()

    # 保留末槽给 LOAD；调用方只分配完整页，碎页不占普通页额度。
    target.bounded_att_big_page_allocator = StatePageAllocator(3, reserved_slots=1)
    target.bounded_att_cpu_load_big_page_slot = 2
    allocator = target.bounded_att_big_page_allocator
    slots = allocator.alloc(token_count // 4)
    original_slots = slots.copy()
    free_count = allocator.free_count
    target.read_req_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=page_ids,
        token_indices=torch.arange(4, 4 + token_count, device="cuda"),
        state_token_len=token_count,
        layouts=layouts,
        big_page_slot_ids=slots,
        req_idx=2,
    )
    torch.cuda.synchronize()
    assert slots == original_slots
    assert allocator.free_count == free_count
    for src, dst in zip(source.get_bounded_attention_states(), target.get_bounded_attention_states()):
        pool = dst.big_pages_pool
        for src_slot, dst_slot in zip(source_slots, slots):
            torch.testing.assert_close(pool.conv_state[dst_slot], src.big_pages_pool.conv_state[src_slot])
            torch.testing.assert_close(pool.ssm_state[dst_slot], src.big_pages_pool.ssm_state[src_slot])
        final_slot = source_slots[page_count - 1]
        torch.testing.assert_close(dst.conv_state[:, 2, :, :2].cpu(), src.big_pages_pool.conv_state[final_slot])
        torch.testing.assert_close(dst.ssm_state[:, 6].cpu(), src.big_pages_pool.ssm_state[final_slot])
        assert not dst.conv_state[:, :2].any()
        if token_count % 4:
            torch.testing.assert_close(dst.big_pages_pool.conv_state[2], src.big_pages_pool.conv_state[final_slot])
            torch.testing.assert_close(dst.big_pages_pool.ssm_state[2], src.big_pages_pool.ssm_state[final_slot])
    for src, dst in zip(source.get_full_attention_states(), target.get_full_attention_states()):
        torch.testing.assert_close(read_kv_storage(dst)[:, 4:4 + token_count], read_kv_storage(src)[:, :token_count])


@pytest.mark.parametrize("tail_pool_size", [4, 8])
def test_cpu_restore_resets_selected_tail_after_all_pages(tail_pool_size: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source, target = composition(device="cuda", tail_pool_size=tail_pool_size), composition(device="cuda", tail_pool_size=tail_pool_size)
    slots = init_source(source)
    layouts = layouts_for(source)
    pages = torch.empty((2, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8, pin_memory=True)
    source.write_req_to_cpu(
        token_indices=torch.arange(8, dtype=torch.long, device="cuda"),
        cpu_page_buffers=pages,
        cpu_page_indices=[0, 1],
        layouts=layouts,
        big_page_slot_ids=slots,
    )
    torch.cuda.synchronize()
    target_slots = target.bounded_att_big_page_allocator.alloc(2)
    tail = getattr(target, f"main_{target.att_tail_type}_state")
    tail.tail_buffer.fill_(37)
    # 缓存流程保证最终前缀 L=8 是可恢复边界；中间页 L=4 不单独重置 tail。
    target.read_req_from_cpu(
        cpu_page_buffers=pages,
        cpu_page_indices=[0, 1],
        token_indices=torch.arange(8, dtype=torch.long, device="cuda"),
        state_token_len=8,
        layouts=layouts,
        big_page_slot_ids=target_slots,
        req_idx=0,
    )
    assert not getattr(target, f"main_{target.att_tail_type}_state").tail_buffer[:, 0].any()
    assert (getattr(target, f"main_{target.att_tail_type}_state").tail_buffer[:, 1:] == 37).all()


@pytest.mark.parametrize("device", ["cuda"])
def test_pd_composition_and_dp_copy_keep_resource_identity(device: str | torch.device | int, pd_args: EasyDict):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source, target, dp_target = composition(device), composition(device), composition(device)
    init_source(source)
    for kind in ("kv", "att_state"):
        layouts = layouts_for(source, kind=kind)
        page = torch.empty(max(layout.byte_end for layout in layouts.values()), dtype=torch.uint8, device=device)
        src_args = {"token_indices": torch.tensor([0, 2], dtype=torch.long, device=device)} if kind == "kv" else {"req_idx": 1, "state_token_len": 8}
        dst_args = {"token_indices": torch.tensor([5, 7], dtype=torch.long, device=device)} if kind == "kv" else {"req_idx": 2, "state_token_len": 8}
        source.write_to_pd_page(
            pd_page_buffers=page.unsqueeze(0),
            pd_page_index=0,
            kind=kind,
            layouts=layouts,
            **src_args,
        )
        target.read_from_pd_page(
            pd_page_buffers=page.unsqueeze(0),
            pd_page_index=0,
            kind=kind,
            layouts=layouts,
            **dst_args,
        )
    dp_tail = dp_target.get_attention_tail_states()[0]
    dp_tail.tail_buffer.fill_(19)
    dp_target.copy_from_dp(
        remote_att_manager=source,
        src_token_indices=torch.tensor([0, 2], dtype=torch.int32, device=device),
        dst_token_indices=torch.tensor([5, 7], dtype=torch.long, device=device),
        src_big_page_ids=[],
        dst_big_page_ids=[],
        src_req_idx=1,
        dst_req_idx=2,
    )
    for a, b in zip(
        (target.main_gqa_fp32_state, target.draft0_mla_fp32_state),
        (dp_target.main_gqa_fp32_state, dp_target.draft0_mla_fp32_state),
    ):
        assert torch.equal(read_kv_storage(a)[:, [5, 7]], read_kv_storage(b)[:, [5, 7]])
    for a, b in zip(
        (target.main_kda_conv_fp32_ssm_fp32_state, target.draft0_gdn_conv_fp32_ssm_fp32_state), (dp_target.main_kda_conv_fp32_ssm_fp32_state, dp_target.draft0_gdn_conv_fp32_ssm_fp32_state)
    ):
        assert torch.equal(a.conv_state[:, 2], b.conv_state[:, 2])
        assert torch.equal(a.ssm_state[:, 6], b.ssm_state[:, 6])
    pd_tail = target.get_attention_tail_states()[0].tail_buffer[:, 2]
    expected_tail = torch.zeros_like(pd_tail)
    for position in range(4, 8):
        expected_tail[:, position % pd_tail.shape[1]].fill_(9)
    assert torch.equal(pd_tail, expected_tail)
    assert not dp_tail.tail_buffer[:, 2].any()
    assert (dp_tail.tail_buffer[:, :2] == 19).all()
    assert (source.get_attention_tail_states()[0].tail_buffer[:, 1] == 9).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bounded transfers")
def test_dp_copy_resets_only_selected_tail_requests(pd_args: EasyDict):
    source, target = composition(device="cuda"), composition(device="cuda")
    init_source(source)
    tails = target.get_attention_tail_states()
    for state in tails:
        state.tail_buffer.fill_(19)
    target.copy_from_dp(
        remote_att_manager=source,
        src_token_indices=torch.empty(0, dtype=torch.long),
        dst_token_indices=torch.empty(0, dtype=torch.long),
        src_big_page_ids=[],
        dst_big_page_ids=[],
        src_req_idx=1,
        dst_req_idx=2,
    )
    assert all(not state.tail_buffer[:, 2].any() for state in tails)
    assert all((state.tail_buffer[:, :2] == 19).all() for state in tails)


@pytest.mark.parametrize("copy_tokens,copy_pages", [(True, True), (True, False), (False, True)])
def test_dp_copy_includes_selected_runtime_and_main_draft_pages(copy_tokens: bool, copy_pages: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda"
    source, target = composition(device), composition(device)
    init_source(source)
    for state in target.get_full_attention_states():
        for buffer in full_state_test_buffers(state).values():
            buffer.fill_(-1)
    for index, state in enumerate(source.get_bounded_attention_states()):
        for page in range(3):
            state.big_pages_pool.conv_state[page].fill_(10 * index + page)
            state.big_pages_pool.ssm_state[page].fill_(20 * index + page)
    for state in target.get_bounded_attention_states():
        state.big_pages_pool.conv_state.fill_(-1)
        state.big_pages_pool.ssm_state.fill_(-2)
        state.small_pages_pool.conv_state.fill_(31)
        state.small_pages_pool.ssm_state.fill_(32)
        state.conv_state.fill_(41)
        state.ssm_state.fill_(42)
        state.mtp_state_index.fill_(1)
    for state in target.get_attention_tail_states():
        state.tail_buffer.fill_(51)

    target.copy_from_dp(
        remote_att_manager=source,
        src_token_indices=torch.tensor([0, 2] if copy_tokens else [], dtype=torch.long, device=device),
        dst_token_indices=torch.tensor([5, 7] if copy_tokens else [], dtype=torch.long, device=device),
        src_big_page_ids=[2, 0] if copy_pages else [],
        dst_big_page_ids=[0, 1] if copy_pages else [],
        src_req_idx=1,
        dst_req_idx=2,
    )

    for state in target.get_full_attention_states():
        src_state = getattr(source, state.state_name)
        values = read_kv_storage(state)
        if copy_tokens:
            assert torch.equal(values[:, [5, 7]], read_kv_storage(src_state)[:, [0, 2]])
        else:
            assert (values == -1).all()
        assert (values[:, :5] == -1).all()
    for state in target.get_bounded_attention_states():
        src_pool = getattr(source, state.state_name).big_pages_pool
        dst_pool = state.big_pages_pool
        if copy_pages:
            for src_page, dst_page in ((2, 0), (0, 1)):
                assert torch.equal(dst_pool.conv_state[dst_page], src_pool.conv_state[src_page])
                assert torch.equal(dst_pool.ssm_state[dst_page], src_pool.ssm_state[src_page])
        else:
            assert (dst_pool.conv_state == -1).all() and (dst_pool.ssm_state == -2).all()
        assert (dst_pool.conv_state[2] == -1).all() and (dst_pool.ssm_state[2] == -2).all()
        assert (state.small_pages_pool.conv_state == 31).all()
        assert (state.small_pages_pool.ssm_state == 32).all()
        source_state = getattr(source, state.state_name)
        assert torch.equal(
            state.conv_state[:, 2, :, :state.config.conv_width],
            source_state.conv_state[:, 1, :, :source_state.config.conv_width],
        )
        dst_slot = 2 * (state.mtp_step + 1)
        src_slot = source_state.mtp_step + 1
        assert torch.equal(state.ssm_state[:, dst_slot], source_state.ssm_state[:, src_slot])
        assert (state.conv_state[:, :2] == 41).all()
        assert (state.conv_state[:, 2, :, state.config.conv_width:] == 41).all()
        assert (state.ssm_state[:, :dst_slot] == 42).all()
        assert (state.ssm_state[:, dst_slot + 1:] == 42).all()
        assert state.mtp_state_index[2] == 0
        assert (state.mtp_state_index[:2] == 1).all()
    for state in target.get_attention_tail_states():
        assert not state.tail_buffer[:, 2].any()
        assert (state.tail_buffer[:, :2] == 51).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("readies", [[True, False, False], [False, True, False], [False, False, True], [True, True, True]])
def test_cpu_ready_joint_pages_preserve_full_and_bounded_fields(readies: list[bool]):
    source = composition("cuda", with_tail=False)
    big_slots = init_source(source)
    small_slots = source.bounded_att_small_page_allocator.alloc(1)
    source.save_bounded_att_runtime_to_pages(
        req_indices=[1],
        state_token_lens=[6],
        page_kind="small",
        page_slot_ids=small_slots,
    )
    torch.cuda.synchronize()
    layouts = layouts_for(source)
    page_ids = [2, 0, 3]
    token_ids = torch.arange(10, device="cuda")
    baseline = torch.full(
        size=(4, max(layout.byte_end for layout in layouts.values())),
        fill_value=139,
        dtype=torch.uint8,
        pin_memory=True,
    )
    pages = torch.full_like(baseline, 139, pin_memory=True)
    source.write_req_to_cpu(
        token_indices=token_ids,
        cpu_page_buffers=baseline,
        cpu_page_indices=page_ids,
        layouts=layouts,
        big_page_slot_ids=big_slots,
        small_page_slot_ids=small_slots,
    )
    torch.cuda.synchronize()
    expected = pages.clone()
    for page, ready in zip(page_ids, readies):
        if not ready:
            expected[page].copy_(baseline[page])
    # ready 页不能访问其页池快照；负的越界槽位可检测是否在构造视图前跳过。
    masked_big_slots = [-999 if readies[i] else slot for i, slot in enumerate(big_slots)]
    masked_small_slots = [-999] if readies[2] else small_slots
    source.write_req_to_cpu(
        token_indices=token_ids,
        cpu_page_buffers=pages,
        cpu_page_indices=page_ids,
        layouts=layouts,
        cpu_page_ready_flags=readies,
        big_page_slot_ids=masked_big_slots,
        small_page_slot_ids=masked_small_slots,
    )
    torch.cuda.synchronize()
    assert torch.equal(pages, expected)
