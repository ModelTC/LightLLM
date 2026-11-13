"""Bounded-memory causal Indexer selection and sparse GQA, prefill/decode.

The kernels can also run with TRITON_INTERPRET=1 on CPU for numerical tests.
No quantization/Hadamard/MLA approximation is applied to the trained Indexer.
"""
import os
import torch
import triton
import triton.language as tl
from ..dsa import query_metadata


@triton.jit
def _scores(
    IQ,
    IW,
    KV,
    Table,
    Req,
    Pos,
    Out,
    iq0: tl.constexpr,
    iq1: tl.constexpr,
    iw0: tl.constexpr,
    kv0: tl.constexpr,
    kv1: tl.constexpr,
    table0: tl.constexpr,
    NK: tl.constexpr,
    NH: tl.constexpr,
    ID: tl.constexpr,
    AD: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BK + tl.arange(0, BK)
    position = tl.load(Pos + row)
    request = tl.load(Req + row)
    score = tl.full((BK,), float("-inf"), tl.float32)
    if tl.program_id(1) * BK <= position:
        valid = (columns < NK) & (columns <= position) & (position >= 0)
        mem = tl.load(Table + request * table0 + columns, valid, other=0).to(tl.int64)
        dims = tl.arange(0, BD)
        keys = tl.load(KV + mem[None, :] * kv0 + AD + dims[:, None], valid[None, :] & (dims[:, None] < ID), other=0)
        heads = tl.arange(0, BH)
        query = tl.load(
            IQ + row * iq0 + heads[:, None] * iq1 + dims[None, :], (heads[:, None] < NH) & (dims[None, :] < ID), other=0
        )
        weights = tl.load(IW + row * iw0 + heads, heads < NH, other=0).to(tl.float32)
        dot = tl.dot(query, keys, input_precision="ieee")
        score = tl.sum(tl.maximum(dot, 0.0) * weights[:, None], 0)
        score = tl.where(valid, score, float("-inf"))
    tl.store(Out + row * NK + columns, score, columns < NK)


@triton.jit
def _sparse_attention(
    Q,
    K,
    V,
    Table,
    Req,
    Indices,
    Out,
    q0: tl.constexpr,
    q1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    table0: tl.constexpr,
    NH: tl.constexpr,
    NKH: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    group = head // (NH // NKH)
    request = tl.load(Req + row)
    dims = tl.arange(0, BD)
    query = tl.load(Q + row * q0 + head * q1 + dims, dims < D, other=0).to(tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    accumulator = tl.full((BD,), 0.0, tl.float32)
    for start in range(tl.cdiv(TOPK, BK)):
        slots = start * BK + tl.arange(0, BK)
        indices = tl.load(Indices + row * TOPK + slots, slots < TOPK, other=-1)
        valid = (slots < TOPK) & (indices >= 0)
        mem = tl.load(Table + request * table0 + indices, valid, other=0).to(tl.int64)
        key = tl.load(
            K + mem[:, None] * k0 + group * k1 + dims[None, :], valid[:, None] & (dims[None, :] < D), other=0
        ).to(tl.float32)
        value = tl.load(
            V + mem[:, None] * v0 + group * v1 + dims[None, :], valid[:, None] & (dims[None, :] < D), other=0
        ).to(tl.float32)
        logits = tl.sum(key * query[None, :], 1) * (D ** -0.5)
        logits = tl.where(valid, logits, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 0))
        # Empty/padded query: avoid (-inf)-(-inf), and return exactly zero.
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.where(valid, tl.exp(logits - safe_max), 0.0)
        accumulator = accumulator * correction + tl.sum(probability[:, None] * value, 0)
        denominator = denominator * correction + tl.sum(probability, 0)
        maximum = new_max
    output = accumulator / tl.maximum(denominator, 1e-20)
    tl.store(Out + row * NH * D + head * D + dims, output, dims < D)


def indexer_scores(iq, iw, kv, table, req_ids, positions, max_keys, attention_dim):
    scores = torch.empty((iq.shape[0], max_keys), dtype=torch.float32, device=iq.device)
    _scores[(iq.shape[0], triton.cdiv(max_keys, 64))](
        iq,
        iw,
        kv,
        table,
        req_ids,
        positions,
        scores,
        iq.stride(0),
        iq.stride(1),
        iw.stride(0),
        kv.stride(0),
        kv.stride(1),
        table.stride(0),
        max_keys,
        iq.shape[1],
        iq.shape[2],
        attention_dim,
        64,
        triton.next_power_of_2(iq.shape[2]),
        max(16, triton.next_power_of_2(iq.shape[1])),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return scores


def selected_attention(q, key, value, table, req_ids, indices):
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _sparse_attention[(q.shape[0], q.shape[1])](
        q,
        key,
        value,
        table,
        req_ids,
        indices,
        output,
        q.stride(0),
        q.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        table.stride(0),
        q.shape[1],
        key.shape[1],
        q.shape[2],
        indices.shape[1],
        32,
        triton.next_power_of_2(q.shape[2]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


SCORE_MEMORY_MIB = int(os.environ.get("DSA_SCORE_MEMORY_MIB", "64"))
TOPK_IMPL = os.environ.get("DSA_TOPK_IMPL", "auto")
ATTN_IMPL = os.environ.get("DSA_SPARSE_ATTN_IMPL", "triton")
# The original packed rank key is exact and materially cheaper for decode
# batches: decode has a small score matrix, so the int64 workspace is bounded,
# while prefill keeps the FP32/TMA path for its large matrices.
DECODE_PACKED = os.environ.get("DSA_DECODE_PACKED", "1") == "1"
SCORE_QUERY_TILE = int(os.environ.get("DSA_SCORE_QUERY_TILE", "8"))
# FlashInfer page TopK has a fixed launch cost on tiny decode batches. The
# exact radix selector plus the existing page-map kernel is faster there;
# explicit DSA_TOPK_IMPL=flashinfer still forces the FlashInfer path.
FLASHINFER_MIN_ROWS = int(os.environ.get("DSA_FLASHINFER_MIN_ROWS", "16"))
PREFILL_TOPK_TIE_REPAIR = os.environ.get("DSA_PREFILL_TOPK_TIE_REPAIR", "1") == "1"
PREFILL_TOPK_TIE_BATCH_ROWS = int(os.environ.get("DSA_PREFILL_TOPK_TIE_BATCH_ROWS", "64"))
if (
    SCORE_MEMORY_MIB < 1
    or TOPK_IMPL not in ("packed", "canonical", "torch", "radix", "vllm", "flashinfer", "auto")
    or ATTN_IMPL not in ("fa3", "triton")
    or SCORE_QUERY_TILE not in (4, 8, 16)
    or FLASHINFER_MIN_ROWS < 1
    or PREFILL_TOPK_TIE_BATCH_ROWS < 1
):
    raise ValueError("Invalid DSA performance options")


def paged_dsa_attention(q, iq, iw, key, value, indexer_key, infer_state, topk, max_keys, *, score_memory_mib=None):
    from .dsa_topk import canonical_topk
    from .dsa_perf import (
        ranked_scores,
        packed_topk,
        fa3_selected_attention,
        flashinfer_topk_page_table,
        flashinfer_topk_page_table_available,
        vllm_prefill_topk_page_table,
        vllm_prefill_topk_available,
        tma_prefill_score_available,
    )
    from .dsa_select import exact_float_topk

    table = infer_state.req_manager.req_to_token_indexs
    requests = getattr(infer_state, "dsa_query_requests", None)
    positions = getattr(infer_state, "dsa_query_positions", None)
    if requests is None or positions is None:
        requests, positions = query_metadata(
            infer_state.b1_cu_q_seq_len, infer_state.b_kv_seq_len, infer_state.b_req_idx, q.shape[0]
        )
    max_keys = max(1, min(max_keys, table.shape[1]))
    packed = (
        TOPK_IMPL == "packed"
        or (DECODE_PACKED and TOPK_IMPL == "auto" and not infer_state.is_prefill)
    ) and iq.shape[1:] == (8, 128)
    memory_mib = SCORE_MEMORY_MIB if score_memory_mib is None else score_memory_mib
    chunk = max(1, min(q.shape[0], memory_mib * 1024 ** 2 // ((8 if packed else 4) * max_keys)))
    if infer_state.is_prefill:
        query_tile = SCORE_QUERY_TILE
    else:
        # Decode batches with at least eight rows amortize the shared Indexer
        # key loads better with an 8-query tile; keep the 4-row path for tiny
        # batches and explicit SCORE_QUERY_TILE=4 tuning runs.
        query_tile = min(SCORE_QUERY_TILE, 8)
        if q.shape[0] < query_tile:
            query_tile = 4
    if packed:
        query_tile = 4
    if chunk >= query_tile:
        chunk = chunk // query_tile * query_tile
    selection_count = min(topk, max_keys)
    # Prefill query positions are often a ramp (the prompt tokens of one
    # request).  Scoring every row against the global max length writes and
    # scans a large -inf suffix.  Process each query chunk at its own causal
    # maximum; decode keeps the reusable fixed-size workspace and graph shape.
    dynamic_prefill = (
        infer_state.is_prefill
        and not packed
        and iq.shape[1:] == (8, 128)
        # For short prefill, allocator/launch overhead outweighs the causal
        # suffix saved by variable shapes.  Keep the fixed path below 256K.
        and max_keys >= 262144
        and not (q.is_cuda and torch.cuda.is_current_stream_capturing())
        # Appended-token prefills already start near max_keys; avoid replacing
        # the reusable workspace when there is no sizable causal suffix.
        and (q.shape[0] == 0 or int(positions[0].item()) + 1 < (max_keys * 3) // 4)
    )
    single_prefill_request = (
        dynamic_prefill
        and getattr(infer_state, "b_req_idx", None) is not None
        and infer_state.b_req_idx.numel() == 1
    )
    score_workspace = None
    if iq.shape[1:] == (8, 128) and not dynamic_prefill:
        score_workspace = torch.empty(
            (chunk, max_keys),
            dtype=torch.int64 if packed else torch.float32,
            device=q.device,
        )
    # vLLM's prefill radix selector is faster for the very large score
    # matrices produced by P. It returns logical IDs, so the wrapper maps
    # them to physical pages and repairs only rows with a tied cutoff.
    use_vllm_page_topk = (
        not packed
        and infer_state.is_prefill
        and iq.shape[1:] == (8, 128)
        and ATTN_IMPL == "fa3"
        and q.is_cuda
        and TOPK_IMPL in ("vllm", "auto")
        and (TOPK_IMPL == "vllm" or q.shape[0] >= FLASHINFER_MIN_ROWS)
        and not (q.is_cuda and torch.cuda.is_current_stream_capturing())
        and vllm_prefill_topk_available()
    )
    use_flashinfer_page_topk = (
        not use_vllm_page_topk
        and not packed
        and iq.shape[1:] == (8, 128)
        and ATTN_IMPL == "fa3"
        and q.is_cuda
        and TOPK_IMPL in ("flashinfer", "auto")
        and (TOPK_IMPL == "flashinfer" or q.shape[0] >= FLASHINFER_MIN_ROWS)
        and flashinfer_topk_page_table_available()
    )
    use_radix_page_topk = (
        not use_vllm_page_topk
        and not use_flashinfer_page_topk
        and not packed
        and iq.shape[1:] == (8, 128)
        and ATTN_IMPL == "fa3"
        and q.is_cuda
        and TOPK_IMPL in ("radix", "auto")
        and (TOPK_IMPL == "radix" or q.shape[0] < FLASHINFER_MIN_ROWS)
    )
    if TOPK_IMPL == "flashinfer" and not use_flashinfer_page_topk:
        raise RuntimeError("DSA_TOPK_IMPL=flashinfer requires FlashInfer top_k_page_table_transform")
    if TOPK_IMPL == "vllm" and infer_state.is_prefill and not use_vllm_page_topk:
        raise RuntimeError("DSA_TOPK_IMPL=vllm requires the vLLM prefill TopK backend")
    selections = torch.empty((q.shape[0], selection_count), dtype=torch.int32, device=q.device)
    lengths = getattr(infer_state, "dsa_query_lengths", None)
    if lengths is None:
        lengths = (positions + 1).clamp(0, max_keys).to(torch.int32).contiguous()

    # A single long prefill request has one logical page map.  Compact its
    # Indexer-K once so the score kernel can use aligned TMA tiles instead of
    # issuing an indirect gather for every query tile.  Invalid tail entries
    # are clamped because causal masking makes them unreachable.
    compact_prefill_key = None
    use_tma_prefill = single_prefill_request and tma_prefill_score_available(iq)
    if use_tma_prefill and indexer_key.ndim == 2 and indexer_key.shape[1] == 128:
        request_id = int(requests[0].item())
        physical = table[request_id, :max_keys].to(torch.long)
        physical = physical.clamp(0, max(0, indexer_key.shape[0] - 1))
        compact_prefill_key = indexer_key.index_select(0, physical).contiguous()

    if dynamic_prefill:
        # Once the causal suffix is bounded per chunk, use larger early
        # chunks to amortize FlashInfer TopK launches.  Tighten the chunk
        # again when its bucket would exceed the score workspace budget.
        score_plan = []
        plan_target = max(chunk, 8192)
        budget_bytes = memory_mib * 1024 ** 2
        start = 0
        while start < q.shape[0]:
            end = min(start + plan_target, q.shape[0])
            while True:
                raw_max_keys = max(1, min(max_keys, int(lengths[start:end].max().item())))
                planned_max_keys = min(max_keys, 1 << (raw_max_keys - 1).bit_length())
                max_rows = max(query_tile, budget_bytes // (4 * planned_max_keys))
                if end - start <= max_rows or end - start <= query_tile:
                    break
                end = start + max_rows
            score_plan.append((start, end, planned_max_keys))
            start = end
    else:
        score_plan = ((start, min(start + chunk, q.shape[0]), max_keys) for start in range(0, q.shape[0], chunk))

    deferred_tie_repairs = {}

    def _flush_tie_repairs(bucket_key):
        bucket = deferred_tie_repairs.pop(bucket_key, None)
        if bucket is None:
            return
        tie_scores = torch.cat(bucket["scores"], dim=0)
        tie_rows = torch.cat(bucket["rows"], dim=0)
        tie_requests = torch.cat(bucket["requests"], dim=0)
        tie_lengths = torch.cat(bucket["lengths"], dim=0)
        tie_count = min(topk, bucket_key)
        repaired = flashinfer_topk_page_table(
            tie_scores,
            table,
            tie_requests,
            tie_lengths,
            tie_count,
        )
        if tie_count == selection_count:
            selections.index_copy_(0, tie_rows, repaired)
        else:
            padded = torch.full(
                (repaired.shape[0], selection_count),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
            padded[:, :tie_count] = repaired
            selections.index_copy_(0, tie_rows, padded)

    for start, end, planned_max_keys in score_plan:
        local_max_keys = planned_max_keys
        local_selection_count = selection_count
        local_score_workspace = score_workspace
        if dynamic_prefill:
            local_selection_count = min(topk, local_max_keys)
            local_score_workspace = torch.empty(
                (end - start, local_max_keys), dtype=torch.float32, device=q.device
            )
        if packed:
            keys = ranked_scores(
                iq[start:end],
                iw[start:end],
                indexer_key,
                table,
                requests[start:end],
                positions[start:end],
                max_keys,
                0,
                out=local_score_workspace[: end - start],
            )
            selected = packed_topk(keys, topk)
            del keys
        elif TOPK_IMPL in ("torch", "radix", "vllm", "flashinfer", "auto") and iq.shape[1:] == (8, 128):
            scores = ranked_scores(
                iq[start:end],
                iw[start:end],
                compact_prefill_key if compact_prefill_key is not None else indexer_key,
                table,
                requests[start:end],
                positions[start:end],
                local_max_keys,
                0,
                packed=False,
                query_tile=query_tile,
                out=local_score_workspace[: end - start],
                use_pdl=use_radix_page_topk,
                prefill_unrolled=single_prefill_request and local_max_keys >= 262144,
                contiguous_kv=compact_prefill_key is not None,
            )
            if use_vllm_page_topk:
                tie_scores = tie_rows = None
                if PREFILL_TOPK_TIE_REPAIR:
                    selected, tie_scores, tie_rows = vllm_prefill_topk_page_table(
                        scores,
                        table,
                        requests[start:end],
                        lengths[start:end],
                        local_selection_count,
                        repair_ties=True,
                        return_ties=True,
                    )
                else:
                    selected = vllm_prefill_topk_page_table(
                        scores,
                        table,
                        requests[start:end],
                        lengths[start:end],
                        local_selection_count,
                        repair_ties=False,
                    )
                if local_selection_count < selection_count:
                    selections[start:end].fill_(-1)
                    selections[start:end, :local_selection_count] = selected
                else:
                    selections[start:end] = selected
                if tie_scores is not None:
                    bucket = deferred_tie_repairs.setdefault(
                        local_max_keys,
                        {"scores": [], "rows": [], "requests": [], "lengths": []},
                    )
                    bucket["scores"].append(tie_scores)
                    bucket["rows"].append(tie_rows + start)
                    bucket["requests"].append(requests[start:end].index_select(0, tie_rows))
                    bucket["lengths"].append(lengths[start:end].index_select(0, tie_rows))
                    if sum(x.shape[0] for x in bucket["rows"]) >= PREFILL_TOPK_TIE_BATCH_ROWS:
                        _flush_tie_repairs(local_max_keys)
                del scores
                if dynamic_prefill:
                    del local_score_workspace
                continue
            if use_flashinfer_page_topk:
                topk_page_table = flashinfer_topk_page_table
                if local_selection_count == selection_count:
                    topk_page_table(
                        scores,
                        table,
                        requests[start:end],
                        lengths[start:end],
                        selection_count,
                        out=selections[start:end],
                    )
                else:
                    local_selected = topk_page_table(
                        scores,
                        table,
                        requests[start:end],
                        lengths[start:end],
                        local_selection_count,
                    )
                    selections[start:end].fill_(-1)
                    selections[start:end, :local_selection_count] = local_selected
                    del local_selected
                del scores
                if dynamic_prefill:
                    del local_score_workspace
                continue
            if use_radix_page_topk:
                from .dsa_radix import exact_radix_topk

                selected = exact_radix_topk(
                    scores,
                    lengths[start:end],
                    local_selection_count,
                    block_size=2048 if local_max_keys <= 131072 else 4096,
                    table=table,
                    requests=requests[start:end],
                    use_pdl=True,
                )
                if local_selection_count < selection_count:
                    selections[start:end].fill_(-1)
                    selections[start:end, :local_selection_count] = selected
                else:
                    selections[start:end] = selected
                del scores
                if dynamic_prefill:
                    del local_score_workspace
                continue
            selected = exact_float_topk(scores, lengths[start:end], local_selection_count, backend=TOPK_IMPL)
            del scores
        else:
            scores = indexer_scores(
                iq[start:end], iw[start:end], indexer_key, table, requests[start:end], positions[start:end], max_keys, 0
            )
            values, selected = scores.topk(local_selection_count, dim=-1, sorted=True)
            selected, _ = canonical_topk(scores, selected, values)
            del scores, values
        if local_selection_count < selection_count:
            selections[start:end].fill_(-1)
            selections[start:end, :local_selection_count] = selected
        else:
            selections[start:end] = selected
        del selected
        if dynamic_prefill:
            del local_score_workspace
    if PREFILL_TOPK_TIE_REPAIR:
        for bucket_key in tuple(deferred_tie_repairs):
            _flush_tie_repairs(bucket_key)
    if score_workspace is not None:
        del score_workspace
    if compact_prefill_key is not None:
        del compact_prefill_key
    physical_page_topk = (
        use_vllm_page_topk
        or use_flashinfer_page_topk
        or use_radix_page_topk
    )
    if ATTN_IMPL == "fa3" and q.is_cuda:
        return fa3_selected_attention(
            q,
            key,
            value,
            table,
            requests,
            selections,
            cu_q=getattr(infer_state, "dsa_cu_q", None),
            physical_pages=physical_page_topk,
            physical_lengths=lengths.clamp_max(selection_count) if physical_page_topk else None,
        )
    return selected_attention(q, key, value, table, requests, selections)
