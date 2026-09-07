from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
    gqa_token_decode_attention_flash_decoding,
)
from lightllm.models.gemma4.triton_kernel.sliding_window_decode import sliding_window_decode_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _int_tensor(values, dtype=torch.int32):
    return torch.tensor(values, device="cuda", dtype=dtype)


def _table_reference(q, kv, req_ids, seq_lengths, q_starts, window, scratch_start):
    # Only the visible suffix matters. Rebase very long sequences to avoid
    # allocating a token table proportional to their virtual token positions.
    indexes = torch.zeros((max(req_ids) + 1, window), device="cuda", dtype=torch.int64)
    visible_lengths = [min(length, window) for length in seq_lengths]
    for req_idx, length, q_start, visible_len in zip(req_ids, seq_lengths, q_starts, visible_lengths):
        positions = torch.arange(length - visible_len, length, device="cuda", dtype=torch.int64)
        indexes[req_idx, :visible_len] = req_idx * window + positions % window
        indexes[req_idx, visible_len - 1] = scratch_start + q_start
    state = SimpleNamespace(
        batch_size=len(req_ids),
        b_req_idx=_int_tensor(req_ids),
        b_seq_len=_int_tensor(visible_lengths),
        max_kv_seq_len=window,
        req_manager=SimpleNamespace(req_to_token_indexs=indexes),
    )
    kv_heads = kv.shape[1] // 2
    return gqa_token_decode_attention_flash_decoding(
        q=q,
        infer_state=state,
        cache_k=kv[:, :kv_heads],
        cache_v=kv[:, kv_heads:],
        out=torch.empty_like(q),
        sliding_window=(window - 1, 0),
    )


@pytest.fixture(autouse=True)
def _use_default_gqa_schedule(monkeypatch):
    # Compare identical math and tiling rather than a machine-specific tune.
    from lightllm.common.triton_utils import autotuner

    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: autotuner.AutotuneLevel.CLOSE_AUTOTUNE)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [512, 1024])
@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(4, 1, 64), (8, 2, 256), (32, 2, 128)])
def test_formula_decode_matches_table_gqa_exactly(dtype, window, q_heads, kv_heads, head_dim):
    torch.manual_seed(42)
    req_ids = [6, 0, 4, 2, 7, 1]
    seq_lengths = [1, 2, window - 1, window, window + 1, 3 * window + 7]
    q_starts = [7, 1, 11, 4, 9, 2]
    scratch_start = 9 * window
    kv = torch.randn((scratch_start + 12, 2 * kv_heads, head_dim), device="cuda", dtype=dtype)
    q = torch.randn((len(req_ids), q_heads, head_dim), device="cuda", dtype=dtype)
    output = torch.empty_like(q)
    actual = sliding_window_decode_attention(
        q,
        kv[:, :kv_heads],
        kv[:, kv_heads:],
        _int_tensor(req_ids),
        _int_tensor(seq_lengths),
        _int_tensor(q_starts),
        window,
        scratch_start,
        out=output,
    )
    expected = _table_reference(q, kv, req_ids, seq_lengths, q_starts, window, scratch_start)
    assert actual is output
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 17, 65])
def test_formula_decode_preserves_gqa_batch_schedule(batch_size):
    window, head_dim, scratch_start = 512, 64, (batch_size + 1) * 512
    req_ids = list(reversed(range(batch_size)))
    seq_lengths = [2 * window + i + 1 for i in range(batch_size)]
    q_starts = list(range(batch_size))
    kv = torch.randn((scratch_start + batch_size, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    q = torch.randn((batch_size, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    actual = sliding_window_decode_attention(
        q,
        kv[:, :1],
        kv[:, 1:],
        _int_tensor(req_ids),
        _int_tensor(seq_lengths),
        _int_tensor(q_starts),
        window,
        scratch_start,
    )
    expected = _table_reference(q, kv, req_ids, seq_lengths, q_starts, window, scratch_start)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_formula_decode_supports_int64_virtual_token_positions():
    window, scratch_start = 512, 4 * 512
    req_ids, seq_lengths, q_starts = [2, 0], [2 ** 31 + 17, 2 ** 32 + 31], [0, 1]
    kv = torch.randn((scratch_start + 2, 2, 64), device="cuda", dtype=torch.bfloat16)
    q = torch.randn((2, 4, 64), device="cuda", dtype=torch.bfloat16)
    actual = sliding_window_decode_attention(
        q,
        kv[:, :1],
        kv[:, 1:],
        _int_tensor(req_ids),
        _int_tensor(seq_lengths, dtype=torch.int64),
        _int_tensor(q_starts),
        window,
        scratch_start,
    )
    expected = _table_reference(q, kv, req_ids, seq_lengths, q_starts, window, scratch_start)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("window", [512, 1024])
def test_formula_decode_cuda_graph_replay_with_changed_requests_and_padding(dtype, window):
    scratch_start = 8 * window
    kv = torch.randn((scratch_start + 4, 4, 256), device="cuda", dtype=dtype)
    q = torch.randn((4, 8, 256), device="cuda", dtype=dtype)
    b_req = _int_tensor([4, 1, 7, 7])
    b_seq = _int_tensor([window + 7, 3, 2, 2])
    b_start = _int_tensor([3, 0, 1, 2])
    out = torch.empty_like(q)

    def forward():
        sliding_window_decode_attention(q, kv[:, :2], kv[:, 2:], b_req, b_seq, b_start, window, scratch_start, out=out)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    req_ids, seq_lengths, q_starts = [2, 5, 7, 7], [2 * window + 3, 1, 2, 2], [1, 3, 2, 0]
    b_req.copy_(_int_tensor(req_ids))
    b_seq.copy_(_int_tensor(seq_lengths))
    b_start.copy_(_int_tensor(q_starts))
    q.mul_(0.5)
    kv.mul_(0.75)
    graph.replay()
    expected = _table_reference(q, kv, req_ids, seq_lengths, q_starts, window, scratch_start)
    # Padding may share the hold request ID; its outputs are intentionally discarded.
    torch.testing.assert_close(out[:2], expected[:2], atol=0, rtol=0)
    assert torch.isfinite(out).all()
