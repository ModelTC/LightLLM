import pytest
import torch

from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm
from lightllm.common.basemodel.triton_kernel.sliding_window_state import build_sliding_window_page_table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _compare_runtime_and_paged(window, q_len, dtype, head_dim=64, image_span=None):
    torch.manual_seed(42)
    req_ids, q_lens = [5, 1, 3], [q_len, 7, 173]
    histories = [2 * window + 73, 0, window - 19]
    lengths = [history + count for history, count in zip(histories, q_lens)]
    # Request IDs and physical query offsets are deliberately unrelated. The
    # flattened query buffer also has gaps, which must not be read or written.
    starts = [11, 47 + q_lens[0], 83 + q_lens[0] + q_lens[1]]
    query_num = max(start + count for start, count in zip(starts, q_lens)) + 13
    kv_heads, q_heads = 2, 4
    req_slots = max(req_ids) + 2
    reference = torch.randn((sum(lengths) + 29, 2 * kv_heads, head_dim), device="cuda", dtype=dtype)
    mapping = torch.full((req_slots, max(lengths) + 11), -1, device="cuda", dtype=torch.int32)
    shuffled_indexes = torch.randperm(reference.shape[0], device="cuda")
    offset = 0
    for req_id, length in zip(req_ids, lengths):
        mapping[req_id, :length] = shuffled_indexes[offset : offset + length].to(torch.int32)
        offset += length

    runtime_start = req_slots * window
    runtime = torch.full((runtime_start + query_num, 2 * kv_heads, head_dim), -3, device="cuda", dtype=dtype)
    for req_id, history, length, start in zip(req_ids, histories, lengths, starts):
        positions = torch.arange(max(0, history - window), history, device="cuda")
        runtime[req_id * window + positions % window] = reference[mapping[req_id, positions].long()]
        runtime[runtime_start + start : runtime_start + start + length - history] = reference[
            mapping[req_id, history:length].long()
        ]

    image_ends = torch.zeros(query_num, device="cuda", dtype=torch.int32)
    if image_span is not None:
        image_start, image_end = image_span
        image_ends[starts[0] + max(0, image_start) : starts[0] + image_end] = histories[0] + image_end
    q = torch.randn((query_num, q_heads, head_dim), device="cuda", dtype=dtype)
    expected, actual = torch.full_like(q, -11), torch.full_like(q, -11)
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    kwargs = dict(
        q=q,
        b_req_idx=int_tensor(req_ids),
        b_start_loc=int_tensor(starts),
        b_seq_len=int_tensor(lengths),
        b_prompt_cache_len=int_tensor(histories),
        max_input_len=max(q_lens),
        b_image_token_end=image_ends,
        sliding_window=(window - 1, 0),
    )
    context_attention_fwd_gemma4_mm(
        k=reference[:, :kv_heads],
        v=reference[:, kv_heads:],
        o=expected,
        req_to_token_indexs=mapping,
        **kwargs,
    )
    indexes = torch.arange(runtime_start, runtime_start + query_num, device="cuda", dtype=torch.int32)
    page_table, kv_start = build_sliding_window_page_table(
        kwargs["b_req_idx"],
        kwargs["b_seq_len"],
        kwargs["b_prompt_cache_len"],
        kwargs["b_start_loc"],
        indexes,
        window,
        max(q_lens),
    )
    context_attention_fwd_gemma4_mm(
        k=runtime[:, :kv_heads],
        v=runtime[:, kv_heads:],
        o=actual,
        req_to_token_indexs=page_table,
        b_kv_start_pos=kv_start,
        **kwargs,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    if image_span is not None:
        # Verify that this case actually exercises bidirectional attention,
        # rather than comparing two paths where the image mask is a no-op.
        causal = torch.full_like(q, -11)
        kwargs["b_image_token_end"] = torch.zeros_like(image_ends)
        context_attention_fwd_gemma4_mm(
            k=reference[:, :kv_heads],
            v=reference[:, kv_heads:],
            o=causal,
            req_to_token_indexs=mapping,
            **kwargs,
        )
        assert not torch.equal(expected, causal)


@pytest.mark.parametrize("window", [512, 1024])
@pytest.mark.parametrize("q_len", [1, 31, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_runtime_prefill_matches_paged_history_and_current_kv(window, q_len, dtype):
    _compare_runtime_and_paged(window, q_len, dtype)


@pytest.mark.parametrize("window", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("image_span", [(17, 43), (31, 197), (-23, 177)])
def test_runtime_prefill_preserves_image_bidirectional_mask(window, dtype, image_span):
    # Production sliding head dimension uses 64-token query tiles. The cases
    # cover an image inside one tile, multiple tiles, and the cached boundary.
    _compare_runtime_and_paged(window, 384, dtype, head_dim=256, image_span=image_span)


@pytest.mark.parametrize("window", [512, 1024])
def test_runtime_prefill_cuda_graph_replay_reads_updated_request_metadata(window):
    torch.manual_seed(43)
    req_slots, head_dim, max_q_len = 6, 64, window + 33
    query_num = 2 * (window + 64) + 97
    runtime_start = req_slots * window
    runtime = torch.randn((runtime_start + query_num, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    indexes = torch.arange(runtime_start, runtime_start + query_num, device="cuda", dtype=torch.int32)
    q = torch.randn((query_num, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    out = torch.full_like(q, -11)
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req = int_tensor([1, 4])
    b_history = int_tensor([window + 3, 0])
    b_seq = int_tensor([2 * window + 20, 31])
    b_start = int_tensor([13, max_q_len + 47])
    image_ends = torch.zeros(query_num, device="cuda", dtype=torch.int32)
    kwargs = dict(
        q=q,
        k=runtime[:, :2],
        v=runtime[:, 2:],
        b_req_idx=b_req,
        b_start_loc=b_start,
        b_seq_len=b_seq,
        b_prompt_cache_len=b_history,
        max_input_len=max_q_len,
        b_image_token_end=image_ends,
        sliding_window=(window - 1, 0),
    )

    def forward():
        page_table, kv_start = build_sliding_window_page_table(
            b_req, b_seq, b_history, b_start, indexes, window, max_q_len
        )
        context_attention_fwd_gemma4_mm(
            o=out,
            req_to_token_indexs=page_table,
            b_kv_start_pos=kv_start,
            **kwargs,
        )

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()

    req_ids, histories, q_lens, starts = [4, 2], [2 * window + 57, window - 5], [max_q_len, 17], [5, max_q_len + 59]
    lengths = [history + count for history, count in zip(histories, q_lens)]
    b_req.copy_(int_tensor(req_ids))
    b_history.copy_(int_tensor(histories))
    b_seq.copy_(int_tensor(lengths))
    b_start.copy_(int_tensor(starts))
    q.mul_(0.5)
    runtime.mul_(0.75)
    out.fill_(-11)
    graph.replay()

    # Materialize a table only for the independent old-path reference, after
    # changing every piece of GPU metadata used by the captured runtime kernel.
    mapping = torch.full((req_slots, max(lengths)), -1, device="cuda", dtype=torch.int32)
    for req_id, history, length, start in zip(req_ids, histories, lengths, starts):
        positions = torch.arange(max(0, history - window), history, device="cuda", dtype=torch.int32)
        mapping[req_id, positions.long()] = req_id * window + positions % window
        mapping[req_id, history:length] = indexes[start : start + length - history]
    expected = torch.full_like(q, -11)
    context_attention_fwd_gemma4_mm(o=expected, req_to_token_indexs=mapping, **kwargs)
    # Include gaps to verify the captured grid respects the new query lengths.
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
