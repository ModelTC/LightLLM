import sys
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.glm5_next import indexer as indexer_module


@pytest.mark.parametrize(
    "rows,max_pools,expected_chunks",
    [
        (60, 1024, [60]),
        (60, 78080, [16, 16, 16, 12]),
        (128, 262272, [4] * 32),
        (19, 262272, [4, 4, 4, 4, 3]),
    ],
)
def test_eager_decode_bounds_page_workspace_and_preserves_row_topk(monkeypatch, rows, max_pools, expected_chunks):
    def run(**capture_settings):
        return _run_mock_decode(monkeypatch, rows, max_pools, **capture_settings)

    chunked, actual_chunks = run()
    reference, graph_chunks = run(capture=True)
    assert actual_chunks == expected_chunks
    assert graph_chunks == [rows]
    assert torch.equal(chunked, reference)
    bytes_per_query = ((max_pools + 63) // 64) * 64 * 132
    assert max(actual_chunks) * bytes_per_query <= 192 * 1024 ** 2
    assert all(size % 4 == 0 for size in actual_chunks[:-1])


@pytest.mark.parametrize("capture", ["capture", "replay", "stream_capture"])
def test_decode_graph_keeps_one_full_workspace(monkeypatch, capture):
    _, sizes = _run_mock_decode(monkeypatch, 60, 78080, **{capture: True})
    assert sizes == [60]


def _run_mock_decode(monkeypatch, rows, max_pools, *, capture=False, replay=False, stream_capture=False):
    indexer = indexer_module.Glm5NextNsaInfer(
        0, {"index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128, "rms_norm_eps": 1e-5, "index_kpool": 4}, 1
    )
    positions = torch.arange(rows)
    q = positions.view(-1, 1, 1)
    weights = (positions * 7).view(-1, 1)
    req_idx = positions // 4
    pool_counts = torch.tensor([0, 1, 511, 512, 513, 639, 640])[positions % 7]
    att_state = SimpleNamespace(q_kv_seq_len=pool_counts * 4 + positions % 4)
    infer_state = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=object()),
        b_req_idx=req_idx,
        is_cuda_graph_capture=capture,
        is_cuda_graph_replay=replay,
    )
    chunks = []
    indexer_k = object()
    columns = torch.arange(640)[None, :]

    def gather(keys, table, batch_req_idx, lengths, pool_capacity):
        assert keys is indexer_k
        assert table is infer_state.req_manager.req_to_token_indexs
        assert pool_capacity == max_pools
        assert lengths.shape == (batch_req_idx.numel(), 1)
        chunks.append(batch_req_idx.numel())
        return keys, batch_req_idx

    def get_metadata(lengths, page_size, num_sms):
        assert page_size == 64 and num_sms == 120
        return lengths

    def mqa_logits(query, keys, row_weights, lengths, block_table, metadata, pool_capacity, *, clean_logits):
        assert keys is indexer_k and lengths is metadata and not clean_logits
        assert pool_capacity == max_pools
        source_rows = query[:, 0, 0, 0]
        assert torch.equal(row_weights[:, 0], weights[source_rows, 0])
        assert torch.equal(block_table, req_idx[source_rows])
        assert torch.equal(lengths[:, 0], pool_counts[source_rows])
        shifts = source_rows + block_table * 13 + row_weights[:, 0]
        scores = ((columns * 131 + shifts[:, None]) % 641).float()
        return scores.masked_fill(columns >= lengths, torch.nan)

    monkeypatch.setattr(indexer_module, "gather_paged_pools", gather)
    monkeypatch.setattr(indexer_module, "HAS_VLLM", False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: stream_capture)
    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_paged_mqa_logits_metadata=get_metadata, get_num_sms=lambda: 120, fp8_paged_mqa_logits=mqa_logits
        ),
    )
    output = indexer._get_decode_indices(q, weights, indexer_k, infer_state, att_state, max_pools)
    valid_output = torch.arange(512)[None, :] < pool_counts[:, None]
    assert (output[~valid_output] == -1).all()
    assert (
        (output[valid_output] >= 0) & (output[valid_output] < pool_counts[:, None].expand_as(output)[valid_output])
    ).all()
    return output, chunks
