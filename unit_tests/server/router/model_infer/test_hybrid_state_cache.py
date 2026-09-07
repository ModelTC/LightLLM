from types import SimpleNamespace

import pytest
import torch

from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.sliding_window_cache_manager import SlidingWindowStateCacheManager
from lightllm.server.router.model_infer.infer_batch import InferenceContext


def _cpu_linear_req_manager(mtp_step):
    manager = object.__new__(ReqManagerForMamba)
    manager.mtp_step = mtp_step
    manager.req_to_conv_state = SimpleNamespace(
        buffer=torch.arange(2 * 3 * 3 * (4 + mtp_step), dtype=torch.float32).reshape(2, 3, 3, 4 + mtp_step)
    )
    manager.req_to_ssm_state = SimpleNamespace(
        buffer=torch.arange(2 * 3 * (mtp_step + 1) * 5, dtype=torch.float32).reshape(2, 3 * (mtp_step + 1), 5)
    )
    manager.req_to_mtp_state_index = torch.full((3,), mtp_step, dtype=torch.int32) if mtp_step else None
    return manager


@pytest.mark.parametrize("mtp_step", [0, 2])
def test_linear_init_clears_entire_request_state_and_resets_mtp_index(mtp_step):
    manager = _cpu_linear_req_manager(mtp_step)
    req = SimpleNamespace(req_idx=1)
    conv = manager.req_to_conv_state.buffer
    ssm = manager.req_to_ssm_state.buffer
    ssm_start = req.req_idx * (mtp_step + 1)
    conv[:, req.req_idx] = float("nan")
    ssm[:, ssm_start : ssm_start + mtp_step + 1] = float("nan")
    expected_conv, expected_ssm = conv.clone(), ssm.clone()
    expected_conv[:, req.req_idx].zero_()
    expected_ssm[:, ssm_start : ssm_start + mtp_step + 1].zero_()

    manager.init_hybrid_attention_state(req=req)

    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm, expected_ssm, atol=0, rtol=0)
    if mtp_step:
        torch.testing.assert_close(
            manager.req_to_mtp_state_index, torch.tensor([mtp_step, 0, mtp_step], dtype=torch.int32)
        )
    else:
        assert manager.req_to_mtp_state_index is None


@pytest.mark.parametrize("mtp_step", [0, 2])
@pytest.mark.parametrize("page_kind", ["big", "small"])
def test_linear_restore_preserves_mtp_conv_tail_and_noncanonical_ssm_rows(mtp_step, page_kind):
    manager = _cpu_linear_req_manager(mtp_step)
    req = SimpleNamespace(req_idx=1, shared_kv_node=SimpleNamespace(small_page_buffer_idx=2))
    conv_pages = torch.arange(3 * 2 * 3 * 4, dtype=torch.float32).reshape(3, 2, 3, 4) + 1000
    ssm_pages = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5) + 2000
    pages = SimpleNamespace(get_state_cache=lambda buffer_idx: (conv_pages[buffer_idx], ssm_pages[buffer_idx]))
    manager.mem_manager = SimpleNamespace(linear_att_big_page_buffers=pages)
    expected_conv = manager.req_to_conv_state.buffer.clone()
    expected_ssm = manager.req_to_ssm_state.buffer.clone()
    expected_conv[:, req.req_idx, ..., :4] = conv_pages[2]
    expected_ssm[:, req.req_idx * (mtp_step + 1)] = ssm_pages[2]

    if page_kind == "big":
        manager.restore_big_page_state(big_page_buffer_idx=2, req=req)
    else:
        manager.restore_small_page_state(req=req, small_page_buffers=pages)

    torch.testing.assert_close(manager.req_to_conv_state.buffer, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(manager.req_to_ssm_state.buffer, expected_ssm, atol=0, rtol=0)
    if mtp_step:
        torch.testing.assert_close(
            manager.req_to_mtp_state_index, torch.tensor([mtp_step, 0, mtp_step], dtype=torch.int32)
        )
    else:
        assert manager.req_to_mtp_state_index is None


@pytest.mark.parametrize("is_hybrid,radix_cache", [(False, object()), (True, None)])
def test_snapshot_without_hybrid_cache_returns_before_reading_requests(is_hybrid, radix_cache):
    context = InferenceContext(is_hybrid_att_mixed_model=is_hybrid, radix_cache=radix_cache)

    # Neither argument supports iteration or len: an early return must not inspect them.
    context.copy_linear_att_state_to_cache_buffer(b_req_idx=object(), reqs=object())


@pytest.mark.parametrize("chunk_end,cache_len", [(17, 32), (768, 544)])
def test_snapshot_outside_cacheable_boundaries_does_not_allocate_or_copy(chunk_end, cache_len):
    context = InferenceContext(is_hybrid_att_mixed_model=True, radix_cache=object())
    context.args = SimpleNamespace(
        linear_att_hash_page_size=32, linear_att_page_block_num=8, disable_chunked_prefill=False
    )
    req = SimpleNamespace(
        req_idx=0,
        get_chuncked_input_token_len=lambda: chunk_end,
        linear_att_cache_len=cache_len,
        linear_att_len_to_big_page_id={},
        tail_linear_att_small_page_buffer_id=None,
    )

    # The radix object has no allocator and req_manager is None, so either access fails.
    context.copy_linear_att_state_to_cache_buffer(b_req_idx=[0], reqs=[req])

    assert req.linear_att_len_to_big_page_id == {}
    assert req.tail_linear_att_small_page_buffer_id is None


def _cpu_state_cache(size, keep_num=0):
    pages = object.__new__(SlidingWindowStateCacheManager)
    pages.size = size
    pages.keep_num = keep_num
    pages.state_cache = torch.empty((size, 2, 4, 2, 4), dtype=torch.float32)
    pages.clear_to_init_state()
    return pages


def test_sliding_big_snapshot_skips_invalid_requests_and_copies_only_selected_page():
    pages = _cpu_state_cache(3)
    manager = object.__new__(ReqManagerForSlidingWindow)
    manager.sliding_window = 4
    manager.mem_manager = SimpleNamespace(linear_att_big_page_buffers=pages)
    manager.req_to_sliding_window = torch.arange(2 * 12 * 2 * 4, dtype=torch.float32).reshape(2, 3, 4, 2, 4)
    expected = manager.req_to_sliding_window[:, 1].clone()

    # Skipped request IDs are deliberately out of range; GPU request IDs must not be read.
    manager.save_big_page_states(b_req_idx=object(), req_indexes=[999, 1, 888], buffer_indexes=[-1, 2, -1])

    torch.testing.assert_close(pages.get_state_cache(2), expected, atol=0, rtol=0)
    assert torch.count_nonzero(pages.state_cache[:2]).item() == 0
    manager.req_to_sliding_window.fill_(-1)
    torch.testing.assert_close(pages.get_state_cache(2), expected, atol=0, rtol=0)


def test_sliding_state_pool_exhaustion_and_released_slot_reuse():
    pages = _cpu_state_cache(2)
    assert pages.alloc_state_cache(3) is None
    assert pages.get_free_cache_num() == 2
    assert pages.alloc_state_cache(2) == [0, 1]
    assert pages.get_used_cache_num() == 2
    assert pages.alloc_one_state_cache() is None

    pages.free_state_cache([1])
    assert pages.get_free_cache_num() == 1
    assert pages.alloc_one_state_cache() == 1
    assert pages.alloc_one_state_cache() is None

    pages.free_state_cache([0, 1])
    assert pages.get_free_cache_num() == 2
    assert pages.get_used_cache_num() == 0


def test_sliding_state_pool_preserves_cpu_transfer_slots():
    pages = _cpu_state_cache(5, keep_num=2)
    assert pages.alloc_state_cache(3) == [0, 1, 2]
    assert pages.alloc_one_state_cache() is None
    for reserved_id in [3, 4]:
        with pytest.raises(AssertionError):
            pages.free_state_cache([reserved_id])
    pages.free_state_cache([0, 1, 2])
    assert pages.get_free_cache_num() == 3
    pages.clear_to_init_state()
    assert pages.alloc_state_cache(3) == [0, 1, 2]
    assert pages.alloc_one_state_cache() is None
