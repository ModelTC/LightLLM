from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
    gqa_token_decode_attention_flash_decoding,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [512, 1024])
@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(4, 1, 64), (8, 2, 256), (32, 2, 128)])
def test_main_gqa_reads_scattered_window_slots(dtype, window, q_heads, kv_heads, head_dim):
    torch.manual_seed(42)
    req_ids = [6, 0, 4, 2, 7, 1]
    lengths = [1, 2, window - 1, window, window + 1, 3 * window + 7]
    full = torch.randn((sum(lengths), 2 * kv_heads, head_dim), device="cuda", dtype=dtype)
    pool = torch.full((len(req_ids) * window + 1, 2 * kv_heads, head_dim), float("nan"), device="cuda", dtype=dtype)
    full_table = torch.zeros((8, max(lengths)), dtype=torch.int32, device="cuda")
    # Unmapped positions point to NaN KV; slot 0 holds the first real request's KV.
    window_table = torch.full_like(full_table, pool.shape[0] - 1)
    shuffled_slots = torch.arange(pool.shape[0] - 1, device="cuda")
    shuffled_slots[1:] = torch.randperm(pool.shape[0] - 2, device="cuda") + 1
    full_offset, window_offset = 0, 0
    for req, length in zip(req_ids, lengths):
        indexes = torch.arange(full_offset, full_offset + length, device="cuda", dtype=torch.int32)
        full_table[req, :length] = indexes
        retained = min(length, window)
        slots = shuffled_slots[window_offset : window_offset + retained]
        pool[slots] = full[indexes[-retained:].long()]
        window_table[req, length - retained : length] = slots.int()
        full_offset += length
        window_offset += retained
    q = torch.randn((len(req_ids), q_heads, head_dim), device="cuda", dtype=dtype)
    state = SimpleNamespace(
        batch_size=len(req_ids),
        b_req_idx=torch.tensor(req_ids, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        max_kv_seq_len=max(lengths),
        req_manager=SimpleNamespace(req_to_token_indexs=full_table),
    )
    expected = gqa_token_decode_attention_flash_decoding(
        q,
        state,
        full[:, :kv_heads],
        full[:, kv_heads:],
        max_len_in_batch=state.max_kv_seq_len,
        out=torch.empty_like(q),
        sliding_window=(window - 1, 0),
    )
    state.req_manager.req_to_token_indexs = window_table
    actual = gqa_token_decode_attention_flash_decoding(
        q,
        state,
        pool[:, :kv_heads],
        pool[:, kv_heads:],
        max_len_in_batch=state.max_kv_seq_len,
        out=torch.empty_like(q),
        sliding_window=(window - 1, 0),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
