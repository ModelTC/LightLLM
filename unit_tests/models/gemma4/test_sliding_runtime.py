from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.req_manager import req_sampling_params, sliding_window
from lightllm.common.state_cache_manager import SlidingWindowCacheConfig
from lightllm.models.gemma4.infer_struct import Gemma4InferStateInfo
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
from lightllm.models.gemma4.model import Gemma4TpPartModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("window", [1, 8, 31])
def test_request_windows_preserve_kv_across_wrap_restore_and_release(monkeypatch, window):
    monkeypatch.setattr(req_sampling_params, "ReqSamplingParamsManager", lambda _: None)
    monkeypatch.setattr(sliding_window, "get_env_start_args", lambda: SimpleNamespace(batch_max_tokens=128))
    monkeypatch.setattr(sliding_window, "get_dp_world_size", lambda: 1)
    layout = SlidingWindowCacheConfig({0: 0}, {1: 0}, window, 1, 4, 1, 4, torch.float32)
    manager = sliding_window.ReqManagerForSlidingWindow(2, 512, None, layout)
    pool = manager.sliding_mem_manager
    req_ids = [manager.alloc(), manager.alloc()]
    assert manager.alloc() is None
    available = pool.size - 2 * window
    assert pool.allocator.can_use_mem_size == available
    lengths = {req_idx: 0 for req_idx in req_ids}
    alloc = Mock(wraps=pool.alloc)
    free = Mock(wraps=pool.free)
    monkeypatch.setattr(pool, "alloc", alloc)
    monkeypatch.setattr(pool, "free", free)

    def assert_window(req_idx):
        length = lengths[req_idx]
        positions = torch.arange(max(0, length - window), length, device="cuda")
        slots = manager.req_to_sliding_window[req_idx, positions].long()
        actual = pool.kv_buffer[0, slots, 0, 0]
        torch.testing.assert_close(actual, (positions + req_idx * 1000).float())

    for req_idx, count in [(req_ids[1], 3), (req_ids[0], 2), (req_ids[0], window * 3 + 5)] + [
        (req_ids[i % 2], 1) for i in range(window * 3 + 5)
    ]:
        old_len = lengths[req_idx]
        alloc.reset_mock()
        free.reset_mock()
        indexes = torch.cat(manager.alloc_sliding_window_indexes(req_idx, count))
        assert indexes.numel() == count
        assert indexes.unique().numel() == count
        positions = torch.arange(old_len, old_len + count, device="cuda")
        manager.req_to_sliding_window[req_idx, positions] = indexes.cuda()
        pool.kv_buffer[0, indexes.long().cuda()] = (positions + req_idx * 1000).float()[:, None, None]
        # Writing current KV must retain the W-1 history tokens needed by its first query.
        history = torch.arange(max(0, old_len - window + 1), old_len, device="cuda")
        old_slots = manager.req_to_sliding_window[req_idx, history].long()
        torch.testing.assert_close(pool.kv_buffer[0, old_slots, 0, 0], (history + req_idx * 1000).float())
        lengths[req_idx] += count
        manager.update_sliding_window(req_idx, lengths[req_idx], indexes)
        if count == 1:
            alloc.assert_not_called()
            free.assert_not_called()
        assert pool.allocator.can_use_mem_size == available
        for other_req in req_ids:
            assert_window(other_req)

    req_idx = req_ids[0]
    cache = manager.create_small_page_cache_manager(1)
    manager.save_state(req_idx, 0, cache)
    torch.cuda.synchronize()
    manager.free_req(req_idx)
    assert manager.alloc() == req_idx
    # Poison the reserved slots so that restore must actually copy checkpoint data.
    reserved = manager._sliding_req_indexes[req_idx].long().cuda()
    pool.kv_buffer[:, reserved] = float("nan")
    alloc.reset_mock()
    free.reset_mock()
    req = SimpleNamespace(
        req_idx=req_idx, cur_kv_len=0, shared_kv_node=SimpleNamespace(node_prefix_total_len=lengths[req_idx])
    )
    manager.restore_state(req, cache, 0)
    torch.cuda.synchronize()
    alloc.assert_not_called()
    free.assert_not_called()
    for other_req in req_ids:
        assert_window(other_req)
        manager.free_req(other_req)
    assert pool.allocator.can_use_mem_size == pool.size
    manager.alloc()
    manager.free_all()
    assert pool.allocator.can_use_mem_size == pool.size
    assert manager.req_list.is_all_free()
    assert torch.all(manager.req_to_sliding_window[manager.HOLD_REQUEST_ID] == pool.HOLD_TOKEN_MEMINDEX)


@pytest.mark.parametrize("is_prefill", [True, False])
@pytest.mark.parametrize("is_sliding", [True, False])
def test_gemma_attention_uses_correct_pool_and_supports_full_head_dim_512(is_prefill, is_sliding):
    torch.manual_seed(42)
    seq_len, window = 29, 8
    q_len = 5 if is_prefill else 1
    head_dim = 64 if is_sliding else 512
    q = torch.randn((q_len, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((seq_len, 4, head_dim), device="cuda", dtype=q.dtype)
    pool = torch.full((seq_len + 1, 4, head_dim), float("nan"), device="cuda", dtype=q.dtype)
    slots = torch.randperm(seq_len, device="cuda")
    pool[slots] = kv
    valid_table = slots.int()[None, :]
    invalid_table = torch.full_like(valid_table, seq_len)
    state = Gemma4InferStateInfo()
    state.is_prefill = is_prefill
    state.b_req_idx = torch.tensor([0], device="cuda", dtype=torch.int32)
    state.b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    state.b_q_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)
    state.b_ready_cache_len = torch.tensor([seq_len - q_len], device="cuda", dtype=torch.int32)
    state.max_q_seq_len = q_len
    state.max_kv_seq_len = seq_len
    state.batch_size = 1
    state.total_token_num = seq_len
    state.req_manager = SimpleNamespace(
        req_to_token_indexs=invalid_table if is_sliding else valid_table,
        req_to_sliding_window=valid_table if is_sliding else invalid_table,
    )
    state.sliding_window_mem_index = slots[-q_len:].int()
    state.mem_manager = SimpleNamespace(
        sliding_kv_buffer=pool[None, :], get_att_input_params=lambda _: (pool[:, :2], pool[:, 2:])
    )
    model = object.__new__(Gemma4TpPartModel)
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: 0)
    model.is_mtp_draft_model = False
    model._init_att_backend()
    model._init_att_backend1()
    if is_prefill:
        state.prefill_att_state = model.prefill_att_backend.create_att_prefill_state(state)
        state.prefill_att_state1 = model.prefill_att_backend1.create_att_prefill_state(state)
    else:
        state.decode_att_state = model.decode_att_backend.create_att_decode_state(state)
        state.decode_att_state1 = model.decode_att_backend1.create_att_decode_state(state)
    state.init_att_state()
    layer = object.__new__(Gemma4TransformerLayerInfer)
    layer.tp_q_head_num_ = 4
    layer.head_dim_ = head_dim
    layer.kv_cache_layer_index_ = 0
    layer.sliding_cache_index_ = 0
    layer.is_sliding = is_sliding
    layer.is_kv_shared_ = False
    layer.sliding_window_ = window
    layer.alloc_tensor = lambda shape, dtype, device="cuda": torch.empty(shape, dtype=dtype, device=device)
    if is_prefill:
        actual = layer._context_attention_kernel(q, kv[-q_len:], state, None)
    else:
        actual = layer._token_attention_kernel(q, state, None)
    query_positions = torch.arange(seq_len - q_len, seq_len, device="cuda")[:, None]
    key_positions = torch.arange(seq_len, device="cuda")[None, :]
    mask = key_positions <= query_positions
    if is_sliding:
        mask &= key_positions > query_positions - window
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0).float(),
            kv[:, :2].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
            kv[:, 2:].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0).float(),
            attn_mask=mask,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
