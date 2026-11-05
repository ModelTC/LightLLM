from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import build_kv_indexes_and_input_ids


@pytest.mark.parametrize("batch", [0, 1, 7, 257])
@pytest.mark.parametrize("gather", [False, True])
def test_fused_decode_matches_independent_indexing(batch, gather):
    table = torch.arange(12 * 2048, device="cuda", dtype=torch.int32).reshape(12, 2048)[:, ::2]
    next_ids = torch.arange(12 * 4, device="cuda", dtype=torch.int64).reshape(12, 4) + 100
    reqs = torch.arange(batch, device="cuda", dtype=torch.int32) % 12
    ends = torch.arange(batch, device="cuda", dtype=torch.int32) % 1023 + 1
    steps = torch.arange(batch, device="cuda", dtype=torch.int32) % 4
    ids = None if gather else torch.full((batch,), 55, dtype=torch.int64, device="cuda")
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=batch,
        b_req_idx=reqs,
        input_ids=ids,
        max_q_seq_len=1,
        b_seq_len=ends,
        b_mtp_index=steps,
        b_is_decode_req=None,
    )
    indexes, result_ids = build_kv_indexes_and_input_ids(model_input, table, next_ids if gather else None)
    assert torch.equal(indexes, table[reqs.long(), ends.long() - 1])
    assert torch.equal(result_ids, next_ids[reqs.long(), steps.long()] if gather else ids)


@pytest.mark.parametrize("max_q", [1, 255, 513, 4097])
def test_fused_mixed_prefill_matches_independent_indexing(max_q):
    lengths = [max_q, 1, min(7, max_q)]
    starts = [0, lengths[0], lengths[0] + lengths[1]]
    ready = [2, 3, 4]
    table = torch.arange(8 * 10000, device="cuda", dtype=torch.int32).reshape(8, 10000)[:, ::2]
    next_ids = torch.arange(8 * 4, device="cuda", dtype=torch.int64).reshape(8, 4) + 900
    ids = torch.arange(sum(lengths), device="cuda", dtype=torch.int64)
    expected_ids = ids.clone()
    expected_ids[starts[1]] = next_ids[3, 2]
    model_input = SimpleNamespace(
        is_prefill=True,
        batch_size=3,
        b_req_idx=torch.tensor([1, 3, 5], device="cuda", dtype=torch.int32),
        input_ids=ids,
        max_q_seq_len=max_q,
        b_seq_len=torch.tensor([r + q for r, q in zip(ready, lengths)], device="cuda", dtype=torch.int32),
        b_ready_cache_len=torch.tensor(ready, device="cuda", dtype=torch.int32),
        b_prefill_start_loc=torch.tensor(starts, device="cuda", dtype=torch.int32),
        b_mtp_index=torch.tensor([0, 2, 0], device="cuda", dtype=torch.int32),
        b_is_decode_req=torch.tensor([False, True, False], device="cuda"),
    )
    expected_indexes = torch.cat([table[req, r : r + q] for req, r, q in zip([1, 3, 5], ready, lengths)])
    indexes, result_ids = build_kv_indexes_and_input_ids(model_input, table, next_ids)
    assert torch.equal(indexes, expected_indexes)
    assert torch.equal(result_ids, expected_ids)


def test_fused_decode_graph_replay_reads_current_tables():
    table = torch.arange(8 * 32, dtype=torch.int32, device="cuda").reshape(8, 32)
    next_ids = torch.arange(8 * 4, dtype=torch.int64, device="cuda").reshape(8, 4)
    reqs = torch.tensor([0, 1, 2, 7], dtype=torch.int32, device="cuda")
    ends = torch.tensor([4, 5, 6, 2], dtype=torch.int32, device="cuda")
    steps = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device="cuda")
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=4,
        b_req_idx=reqs,
        input_ids=None,
        max_q_seq_len=1,
        b_seq_len=ends,
        b_mtp_index=steps,
        b_is_decode_req=None,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            build_kv_indexes_and_input_ids(model_input, table, next_ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        indexes, ids = build_kv_indexes_and_input_ids(model_input, table, next_ids)
    for iteration in range(3):
        table.add_(1000)
        next_ids.add_(100)
        ends.add_(1)
        graph.replay()
        assert torch.equal(indexes, table[reqs.long(), ends.long() - 1])
        assert torch.equal(ids, next_ids[reqs.long(), steps.long()])
