"""Sparse MLA step builders: per-step lengths and the storage kept for CUDA graph replays."""

import pytest
import torch

from lightllm.common.attention.backend import AttentionMetadata
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 2048


def make_batch(lengths, *, capacity=None, microbatch=0):
    cu = torch.arange(len(lengths) + 1, device=DEVICE, dtype=torch.int32)
    return AttentionMetadata(
        req_to_token_indexs=torch.zeros((len(lengths), 1), device=DEVICE, dtype=torch.int32),
        b_req_idx=torch.arange(len(lengths), device=DEVICE, dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device=DEVICE, dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        cu_q_seq_len=cu,
        max_q_seq_len=1,
        cuda_graph_max_kv_seq_len=capacity,
        microbatch_index=microbatch,
    )


def expected_lengths(lengths, kpool):
    # The indexer selects whole kpool pools within the top-k width plus the newest partial pool.
    selected = [min(n // kpool * kpool, TOPK) + n % kpool for n in lengths]
    prefix = [0]
    for n in selected:
        prefix.append(prefix[-1] + n)
    as_tensor = lambda values: torch.tensor(values, device=DEVICE, dtype=torch.int32)  # noqa: E731
    return as_tensor(selected), as_tensor(prefix)


def assert_att_state(att_state, lengths, kpool):
    cache_seqlens, cu_seqlens_k = expected_lengths(lengths, kpool)
    assert att_state.cu_seqlens_k.dtype == torch.int32
    torch.testing.assert_close(att_state.cache_seqlens.int(), cache_seqlens, rtol=0, atol=0)
    torch.testing.assert_close(att_state.cu_seqlens_k, cu_seqlens_k, rtol=0, atol=0)


@pytest.mark.parametrize("kpool", [1, 4])
def test_fa3_sparse_decode_lengths_follow_the_indexer_selection(kpool):
    builder = FA3SparseMLADecodeBackend.get_step_builder_cls()(topk=TOPK, kpool=kpool, causal=kpool == 1)
    lengths = [1, 3, 5, 2047, 2048, 2049, 2051, 2052, 9000]
    att_state = builder.build(make_batch(lengths))
    assert_att_state(att_state, lengths, kpool)
    assert att_state.causal == (kpool == 1) and att_state.max_seqlen_q == 1


@pytest.mark.parametrize("kpool", [1, 4])
def test_fa3_sparse_decode_replay_rewrites_captured_storage_in_place(kpool):
    builder = FA3SparseMLADecodeBackend.get_step_builder_cls()(topk=TOPK, kpool=kpool)
    captured_lengths = [7, 2049, 4096, 2]
    captured = builder.build(make_batch(captured_lengths, capacity=8192))
    assert_att_state(captured, captured_lengths, kpool)
    pointers = (captured.cache_seqlens.data_ptr(), captured.cu_seqlens_k.data_ptr())

    # A replay step of the same (microbatch, batch size) writes its lengths into the captured storage.
    replay_lengths = [3000, 5, 2051, 64]
    replay = builder.build(make_batch(replay_lengths, capacity=8192))
    assert (replay.cache_seqlens.data_ptr(), replay.cu_seqlens_k.data_ptr()) == pointers
    assert_att_state(captured, replay_lengths, kpool)

    # Eager steps never touch the captured storage.
    eager = builder.build(make_batch([1, 1, 1, 1]))
    assert eager.cache_seqlens.data_ptr() not in pointers and eager.cu_seqlens_k.data_ptr() not in pointers
    assert_att_state(captured, replay_lengths, kpool)


def test_fa3_sparse_decode_keeps_storage_per_microbatch_and_batch_size():
    builder = FA3SparseMLADecodeBackend.get_step_builder_cls()(topk=TOPK, kpool=4)
    first = builder.build(make_batch([9, 10], capacity=8192, microbatch=0))
    second = builder.build(make_batch([2049, 4096], capacity=8192, microbatch=1))
    wider = builder.build(make_batch([9, 10, 11], capacity=8192, microbatch=0))
    tensors = [t for att_state in (first, second, wider) for t in (att_state.cache_seqlens, att_state.cu_seqlens_k)]
    storage = {t.data_ptr() for t in tensors}
    assert len(storage) == 6
    # Building microbatch 1 must not overwrite the lengths microbatch 0's graph reads.
    assert_att_state(first, [9, 10], 4)
    assert_att_state(second, [2049, 4096], 4)
    assert_att_state(wider, [9, 10, 11], 4)
