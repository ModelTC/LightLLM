"""Sparse MLA decode state: per-step lengths and the storage kept for CUDA graph replays."""

import sys
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig, SWASparseMLALayerConfig
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend
from lightllm.common.attention.backends.sparse_mla.flashmla_swa import FlashMLASWADecodeBackend
from lightllm.common.basemodel.infer_struct import GraphMode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 2048


def test_flashmla_swa_allocates_separate_metadata_only_for_capture(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_mla", SimpleNamespace(get_mla_metadata=lambda: (object(), None)))
    config = SWASparseMLALayerConfig(
        num_heads=64, head_dim_v=512, softmax_scale=0.1, dtype=torch.bfloat16, compress_ratio=4
    )
    backend = FlashMLASWADecodeBackend(config)
    for mode in (GraphMode.EAGER, GraphMode.CAPTURE, GraphMode.REPLAY):
        state = backend.build(SimpleNamespace(graph_mode=mode))
        if mode is GraphMode.CAPTURE:
            assert state.capture_sched_meta is not state.sched_meta
        else:
            assert state.capture_sched_meta is None


def fa3_decode(kpool, causal=True):
    config = SparseMLALayerConfig(
        num_heads=16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.1,
        dtype=torch.bfloat16,
        topk=TOPK,
        kpool=kpool,
        causal=causal,
    )
    return FA3SparseMLADecodeBackend(config)


def make_batch(lengths, *, capacity=None, microbatch=0):
    cu = torch.arange(len(lengths) + 1, device=DEVICE, dtype=torch.int32)
    return make_infer_state(
        req_to_token_indexs=torch.zeros((len(lengths), 1), device=DEVICE, dtype=torch.int32),
        b_req_idx=torch.arange(len(lengths), device=DEVICE, dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device=DEVICE, dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        b1_cu_q_seq_len=cu,
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
    lengths = [1, 3, 5, 2047, 2048, 2049, 2051, 2052, 9000]
    att_state = fa3_decode(kpool, causal=kpool == 1).build(make_batch(lengths))
    assert_att_state(att_state, lengths, kpool)
    assert att_state.max_seqlen_q == 1


@pytest.mark.parametrize("kpool", [1, 4])
def test_fa3_sparse_decode_replay_rewrites_captured_storage_in_place(kpool):
    backend = fa3_decode(kpool)
    captured_lengths = [7, 2049, 4096, 2]
    captured = backend.build(make_batch(captured_lengths, capacity=8192))
    assert_att_state(captured, captured_lengths, kpool)
    pointers = (captured.cache_seqlens.data_ptr(), captured.cu_seqlens_k.data_ptr())

    # A replay step of the same (microbatch, batch size) writes its lengths into the captured storage.
    replay_lengths = [3000, 5, 2051, 64]
    replay = backend.build(make_batch(replay_lengths, capacity=8192))
    assert (replay.cache_seqlens.data_ptr(), replay.cu_seqlens_k.data_ptr()) == pointers
    assert_att_state(captured, replay_lengths, kpool)

    # Eager steps never touch the captured storage.
    eager = backend.build(make_batch([1, 1, 1, 1]))
    assert eager.cache_seqlens.data_ptr() not in pointers and eager.cu_seqlens_k.data_ptr() not in pointers
    assert_att_state(captured, replay_lengths, kpool)


def test_fa3_sparse_decode_keeps_storage_per_microbatch_and_batch_size():
    backend = fa3_decode(kpool=4)
    first = backend.build(make_batch([9, 10], capacity=8192, microbatch=0))
    second = backend.build(make_batch([2049, 4096], capacity=8192, microbatch=1))
    wider = backend.build(make_batch([9, 10, 11], capacity=8192, microbatch=0))
    tensors = [t for att_state in (first, second, wider) for t in (att_state.cache_seqlens, att_state.cu_seqlens_k)]
    storage = {t.data_ptr() for t in tensors}
    assert len(storage) == 6
    # Building microbatch 1 must not overwrite the lengths microbatch 0's graph reads.
    assert_att_state(first, [9, 10], 4)
    assert_att_state(second, [2049, 4096], 4)
    assert_att_state(wider, [9, 10, 11], 4)
