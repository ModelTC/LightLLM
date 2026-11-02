from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.flashinfer.paged_metadata import build_page_metadata, prepare_page_metadata


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 7, 96, 257])
@pytest.mark.parametrize("page_size", [1, 16, 256])
def test_fused_metadata_and_capture_follow_new_lengths_and_pages(batch, page_size):
    torch.manual_seed(41)
    max_length = 1024
    request_ids = torch.randperm(batch, device="cuda", dtype=torch.int32)
    lengths = torch.randint(1, max_length + 1, (batch,), device="cuda", dtype=torch.int32)
    page_count = (max_length + page_size - 1) // page_size
    physical_pages = torch.randperm(batch * page_count, device="cuda", dtype=torch.int32).view(batch, page_count)
    table = (physical_pages[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(batch, -1).int()
    infer = SimpleNamespace(
        b_req_idx=request_ids,
        b_seq_len=lengths,
        max_kv_seq_len=max_length,
        req_manager=SimpleNamespace(req_to_token_indexs=table),
    )
    indices = torch.empty(batch * page_count, dtype=torch.int32, device="cuda")
    build_page_metadata(infer, page_size, indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_page_metadata(infer, page_size, indices)
    for step in range(3):
        lengths.copy_(torch.randint(1, max_length + 1, lengths.shape, device="cuda", dtype=torch.int32))
        table.add_(page_size)
        graph.replay()
        pages = (lengths + page_size - 1) // page_size
        expected_starts = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), pages.cumsum(0).int()))
        torch.testing.assert_close(actual.starts, expected_starts, rtol=0, atol=0)
        torch.testing.assert_close(actual.last_page_len, lengths - (pages - 1) * page_size, rtol=0, atol=0)
        expected_indices = torch.cat(
            [table[r, :n:page_size] // page_size for r, n in zip(request_ids.tolist(), lengths.tolist())]
        )
        torch.testing.assert_close(actual.indices[: expected_indices.numel()], expected_indices, rtol=0, atol=0)


def test_layer_shapes_share_page_mapping_but_execution_lanes_do_not():
    infer = SimpleNamespace(
        b_req_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([4, 7], device="cuda", dtype=torch.int32),
        input_ids=torch.zeros(2, device="cuda"),
        batch_size=2,
        is_prefill=False,
        max_kv_seq_len=7,
        microbatch_index=0,
        req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(16, device="cuda").view(2, 8)),
    )
    model = SimpleNamespace(graph_max_batch_size=2, graph_max_len_in_batch=8)

    def backend(heads):
        return SimpleNamespace(
            tp_q_head_num=heads,
            model=model,
            infer_page_size=4,
            max_page_num=2,
            kv_indices_buffer=[torch.empty(4, dtype=torch.int32, device="cuda") for _ in range(2)],
        )

    first = prepare_page_metadata(backend(8), infer)
    assert first is prepare_page_metadata(backend(16), infer)
    infer.microbatch_index = 1
    second = prepare_page_metadata(backend(8), infer)
    assert first.indices.data_ptr() != second.indices.data_ptr()


@pytest.mark.parametrize("is_prefill", [False, True])
def test_dynamic_lengths_cannot_overflow_fixed_graph_page_indices(is_prefill):
    fixed = torch.full((4,), -7, dtype=torch.int32, device="cuda")
    backend = SimpleNamespace(
        model=SimpleNamespace(graph_max_batch_size=2, graph_max_len_in_batch=16),
        infer_page_size=4,
        max_page_num=2,
        kv_indices_buffer=[fixed],
    )
    infer = SimpleNamespace(
        b_req_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([4, 13], device="cuda", dtype=torch.int32),
        input_ids=torch.zeros(2, device="cuda"),
        batch_size=2,
        is_prefill=is_prefill,
        max_kv_seq_len=13,
        microbatch_index=0,
        req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(32, device="cuda").view(2, 16)),
    )
    metadata = prepare_page_metadata(backend, infer)
    assert metadata.starts.tolist() == [0, 1, 5]
    assert metadata.indices[:5].tolist() == [0, 4, 5, 6, 7]
    assert fixed.tolist() == [-7] * 4

    backend.max_page_num = 4
    backend.kv_indices_buffer = [torch.empty(8, dtype=torch.int32, device="cuda")]
    for lengths in ([4, 7], [8, 13]):
        infer.b_seq_len.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        infer.max_kv_seq_len = max(lengths)
        del infer._paged_attention_metadata
        metadata = prepare_page_metadata(backend, infer)
        assert metadata.indices.numel() == 8
