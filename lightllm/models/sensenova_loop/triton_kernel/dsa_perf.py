"""Exact packed top-k, query-tiled scoring, and selected physical KV pages."""
import os

import torch
import triton
import triton.language as tl

INDEX_BITS = 20  # Includes the 512K context and MTP headroom.


_TMA_SCORE_ENABLED = os.environ.get("DSA_PREFILL_SCORE_TMA", "1") == "1"
_TMA_ALLOCATOR_INSTALLED = False


def _install_tma_allocator():
    """Install Triton's small descriptor allocator without touching model tensors."""
    global _TMA_ALLOCATOR_INSTALLED
    if _TMA_ALLOCATOR_INSTALLED or not hasattr(triton, "set_allocator"):
        return

    def _allocator(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    try:
        triton.set_allocator(_allocator)
        _TMA_ALLOCATOR_INSTALLED = True
    except (RuntimeError, TypeError):
        # Older Triton builds may not expose a CUDA descriptor allocator.
        # The pointer-based score kernel remains the exact fallback.
        pass


def tma_prefill_score_available(tensor):
    if not _TMA_SCORE_ENABLED or tensor.device.type != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(tensor.device)[0] >= 9 and hasattr(triton, "set_allocator")
    except RuntimeError:
        return False


_FLASHINFER_TOPK_STATE = None


def flashinfer_topk_page_table_available():
    """Whether FlashInfer provides the exact fused TopK/page-table path."""
    global _FLASHINFER_TOPK_STATE
    if _FLASHINFER_TOPK_STATE is not None:
        return _FLASHINFER_TOPK_STATE
    try:
        import flashinfer

        _FLASHINFER_TOPK_STATE = callable(getattr(flashinfer, "top_k_page_table_transform", None))
    except (ImportError, OSError, RuntimeError):
        _FLASHINFER_TOPK_STATE = False
    return _FLASHINFER_TOPK_STATE


def flashinfer_topk_page_table(scores, table, requests, lengths, topk, *, out=None):
    """Exact FP32 TopK fused with logical-to-physical token mapping.

    FlashInfer's deterministic SMALL tie-break matches the baseline's
    stable descending score order followed by increasing token IDs. The
    output is already in physical cache-slot IDs, so no second gather kernel
    is needed before FA3.
    """
    if not flashinfer_topk_page_table_available():
        raise RuntimeError("FlashInfer fused TopK/page-table backend is unavailable")
    import flashinfer
    from flashinfer import TopKTieBreak

    if requests.dtype != torch.int32:
        requests = requests.to(torch.int32)
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    if not requests.is_contiguous():
        requests = requests.contiguous()
    if not lengths.is_contiguous():
        lengths = lengths.contiguous()
    if not table.is_contiguous():
        table = table.contiguous()
    return flashinfer.top_k_page_table_transform(
        scores,
        table,
        lengths,
        min(topk, scores.shape[1]),
        row_to_batch=requests,
        deterministic=True,
        tie_break=TopKTieBreak.SMALL,
        out=out,
    )


_VLLM_PREFILL_TOPK_STATE = None


def vllm_prefill_topk_available():
    """Whether vLLM exposes the long-prefill radix TopK operator."""
    global _VLLM_PREFILL_TOPK_STATE
    if _VLLM_PREFILL_TOPK_STATE is not None:
        return _VLLM_PREFILL_TOPK_STATE
    try:
        from vllm import _custom_ops as vllm_ops

        _VLLM_PREFILL_TOPK_STATE = callable(getattr(vllm_ops, "top_k_per_row_prefill", None))
    except (ImportError, OSError, RuntimeError):
        _VLLM_PREFILL_TOPK_STATE = False
    return _VLLM_PREFILL_TOPK_STATE


def vllm_prefill_topk(scores, lengths, topk):
    """Run vLLM's radix prefill selector and return logical token IDs."""
    if not vllm_prefill_topk_available():
        raise RuntimeError("vLLM prefill TopK backend is unavailable")
    from vllm import _custom_ops as vllm_ops

    if scores.ndim != 2 or scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("vLLM prefill TopK requires a two-dimensional floating-point score matrix")
    lengths = lengths.to(dtype=torch.int32, device=scores.device).contiguous()
    count = min(topk, scores.shape[1])
    starts = torch.zeros_like(lengths)
    logical = torch.empty((scores.shape[0], count), dtype=torch.int32, device=scores.device)
    vllm_ops.top_k_per_row_prefill(
        scores,
        starts,
        lengths.clamp_max(scores.shape[1]),
        logical,
        scores.shape[0],
        scores.stride(0),
        scores.stride(1),
        count,
    )
    return logical


def vllm_prefill_topk_page_table(
    scores,
    table,
    requests,
    lengths,
    topk,
    *,
    repair_ties=True,
    return_ties=False,
):
    """Select logical IDs with vLLM, map them to physical pages, and repair ties.

    vLLM's prefill radix selector is substantially faster than the FlashInfer
    page transform for long score matrices, but its radix tie order is
    unspecified. The common no-tie case stays on the fast path. With
    return_ties=True the caller receives the small tied-row score subset so
    several chunks can be repaired in one FlashInfer launch.
    """
    logical = vllm_prefill_topk(scores, lengths, topk)
    pages, _ = selected_pages(logical, table, requests)
    if not repair_ties or logical.shape[0] == 0 or not flashinfer_topk_page_table_available():
        return (pages, None, None) if return_ties else pages

    valid = (logical >= 0) & (logical < scores.shape[1])
    valid &= logical < lengths[:, None]
    values = scores.gather(1, logical.clamp_min(0)).masked_fill(~valid, float("-inf"))
    cutoff = values.min(dim=1).values
    tie_rows = ((values == cutoff[:, None]) & valid).sum(dim=1) > 1
    tie_rows &= cutoff > float("-inf")
    row_ids = torch.nonzero(tie_rows, as_tuple=False).flatten()
    if row_ids.numel() != 0:
        if return_ties:
            return pages, scores.index_select(0, row_ids).contiguous(), row_ids
        repaired = flashinfer_topk_page_table(
            scores.index_select(0, row_ids),
            table,
            requests.index_select(0, row_ids),
            lengths.index_select(0, row_ids),
            min(topk, scores.shape[1]),
        )
        pages.index_copy_(0, row_ids, repaired)
    return (pages, None, None) if return_ties else pages


SCORE_KEY_TILE = int(os.environ.get("DSA_SCORE_KEY_TILE", "128"))
if SCORE_KEY_TILE not in (64, 128, 256, 512):
    raise ValueError("DSA_SCORE_KEY_TILE must be 64, 128, 256 or 512")

# The long-prefill TMA kernel reuses each query tile over this many key
# blocks. H200 tuning favors 16: it cuts launch/program overhead while keeping
# the register footprint below the spill point seen at 32. Set 8 to reproduce
# the earlier baseline during A/B experiments.
TMA_PREFILL_KLOOP = int(os.environ.get("DSA_PREFILL_TMA_KLOOP", "16"))
if TMA_PREFILL_KLOOP not in (8, 16, 32):
    raise ValueError("DSA_PREFILL_TMA_KLOOP must be 8, 16 or 32")

# FA3 split-KV tuning for selected-token decode. ``auto`` uses more splits
# for tiny batches, where a single split leaves too little parallel work; the
# value remains overrideable for other GPU generations and cache lengths.
FA3_NUM_SPLITS = os.environ.get("DSA_FA3_NUM_SPLITS", "auto")
if FA3_NUM_SPLITS != "auto":
    try:
        if int(FA3_NUM_SPLITS) < 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError("DSA_FA3_NUM_SPLITS must be auto or a nonnegative integer") from exc


def _fa3_num_splits(num_queries):
    if FA3_NUM_SPLITS != "auto":
        return int(FA3_NUM_SPLITS)
    if num_queries <= 32:
        return 8
    if num_queries <= 256:
        return 1
    return 0


@triton.jit
def _ranking_key(score, column):
    # Order-preserving float32 -> signed integer mapping; +/-zero are equal.
    # Low bits rank lower token indices first, without perturbing the score.
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits < 0, -2147483648 - bits, bits)
    rank = ordered * 1048576 + (1048575 - column.to(tl.int64))
    finite = (score > float("-inf")) & (score < float("inf"))
    return tl.where(finite, rank, tl.full(score.shape, -9223372036854775808, tl.int64))


@triton.jit
def _score_tiled(
    IQ,
    IW,
    KV,
    Table,
    Req,
    Pos,
    Out,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    IQ0: tl.constexpr,
    IQ1: tl.constexpr,
    IW0: tl.constexpr,
    KV0: tl.constexpr,
    TS: tl.constexpr,
    AD: tl.constexpr,
    M: tl.constexpr = 4,
    BN: tl.constexpr = 128,
    PACKED: tl.constexpr = True,
    USE_PDL: tl.constexpr = False,
):
    first = tl.program_id(0) * M
    row = first + tl.arange(0, M)
    req = tl.load(Req + row, row < NQ, other=-1)
    pos = tl.load(Pos + row, row < NQ, other=-1)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    dims = tl.arange(0, 128)
    low = tl.min(tl.where(row < NQ, req, 2147483647), 0)
    high = tl.max(req, 0)
    if low == high:
        # Prefill and fixed-size MTP groups usually share one request. Load
        # its Indexer keys once for four queries and use full Tensor Core tiles.
        if tl.program_id(1) * BN <= tl.max(pos, 0):
            keep_k = (col < NK) & (col <= tl.max(pos, 0))
            mem = tl.load(Table + low * TS + col, keep_k, other=0).to(tl.int64)
            keys = tl.load(KV + mem[None, :] * KV0 + AD + dims[:, None], keep_k[None, :], other=0)
            qh = tl.arange(0, M * 8)
            qrow = first + qh // 8
            query = tl.load(
                IQ + qrow[:, None] * IQ0 + (qh % 8)[:, None] * IQ1 + dims[None, :], qrow[:, None] < NQ, other=0
            )
            weights = tl.load(IW + qrow * IW0 + qh % 8, qrow < NQ, other=0).to(tl.float32)
            dot = tl.dot(query, keys, input_precision="ieee")
            score = tl.sum(tl.reshape(tl.maximum(dot, 0.0) * weights[:, None], (M, 8, BN)), 1)
            keep = (row[:, None] < NQ) & (col[None, :] < NK) & (col[None, :] <= pos[:, None])
            score = tl.where(keep, score, float("-inf"))
        else:
            score = tl.full((M, BN), float("-inf"), tl.float32)
        if PACKED:
            packed = _ranking_key(score, tl.broadcast_to(col[None, :], (M, BN)))
        else:
            packed = tl.where((score > float("-inf")) & (score < float("inf")), score, float("-inf"))
        tl.store(Out + row[:, None] * NK + col[None, :], packed, (row[:, None] < NQ) & (col[None, :] < NK))
    else:
        # Boundary tiles may contain several requests. Score each row against
        # its own table, never borrowing keys from the adjacent request.
        heads = tl.arange(0, 16)
        for offset in range(M):
            qr = first + offset
            request = tl.load(Req + qr, qr < NQ, other=0)
            position = tl.load(Pos + qr, qr < NQ, other=-1)
            score = tl.full((BN,), float("-inf"), tl.float32)
            if (qr < NQ) & (tl.program_id(1) * BN <= position):
                keep = (col < NK) & (col <= position)
                mem = tl.load(Table + request * TS + col, keep, other=0).to(tl.int64)
                keys = tl.load(KV + mem[None, :] * KV0 + AD + dims[:, None], keep[None, :], other=0)
                query = tl.load(IQ + qr * IQ0 + heads[:, None] * IQ1 + dims[None, :], heads[:, None] < 8, other=0)
                weights = tl.load(IW + qr * IW0 + heads, heads < 8, other=0).to(tl.float32)
                score = tl.sum(tl.maximum(tl.dot(query, keys, input_precision="ieee"), 0.0) * weights[:, None], 0)
                score = tl.where(keep, score, float("-inf"))
            if PACKED:
                score = _ranking_key(score, col)
            else:
                score = tl.where((score > float("-inf")) & (score < float("inf")), score, float("-inf"))
            tl.store(Out + qr * NK + col, score, (qr < NQ) & (col < NK))
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _score_tiled_unrolled(
    IQ,
    IW,
    KV,
    Table,
    Req,
    Pos,
    Out,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    IQ0: tl.constexpr,
    IQ1: tl.constexpr,
    IW0: tl.constexpr,
    KV0: tl.constexpr,
    TS: tl.constexpr,
    M: tl.constexpr = 16,
    BN: tl.constexpr = 128,
    KLOOP: tl.constexpr = 8,
    USE_PDL: tl.constexpr = False,
):
    """Score one request while reusing each query tile across key tiles."""
    first = tl.program_id(0) * M
    row = first + tl.arange(0, M)
    row_offset = row.to(tl.int64) * NK
    pos = tl.load(Pos + row, row < NQ, other=-1)
    request = tl.load(Req + first, first < NQ, other=0)
    qh = tl.arange(0, M * 8)
    qrow = first + qh // 8
    dims = tl.arange(0, 128)
    valid_q = qrow < NQ
    query = tl.load(
        IQ + qrow[:, None] * IQ0 + (qh % 8)[:, None] * IQ1 + dims[None, :],
        valid_q[:, None],
        other=0,
    )
    weights = tl.load(IW + qrow * IW0 + qh % 8, valid_q, other=0).to(tl.float32)
    base = tl.program_id(1) * (BN * KLOOP)
    max_pos = tl.max(pos, 0)
    for kt in tl.static_range(0, KLOOP):
        col = base + kt * BN + tl.arange(0, BN)
        keep_k = col < NK
        if base + kt * BN <= max_pos:
            mem = tl.load(Table + request * TS + col, keep_k, other=0).to(tl.int64)
            keys = tl.load(KV + mem[None, :] * KV0 + dims[:, None], keep_k[None, :], other=0)
            dot = tl.dot(query, keys, input_precision="ieee")
            score = tl.sum(tl.reshape(tl.maximum(dot, 0.0) * weights[:, None], (M, 8, BN)), 1)
            keep = (row[:, None] < NQ) & keep_k[None, :] & (col[None, :] <= pos[:, None])
            score = tl.where(keep, score, float("-inf"))
        else:
            score = tl.full((M, BN), float("-inf"), tl.float32)
        tl.store(Out + row_offset[:, None] + col[None, :], score, (row[:, None] < NQ) & keep_k[None, :])
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _score_tiled_unrolled_tma(
    IQ,
    IW,
    KV,
    Pos,
    Out,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    IQ0: tl.constexpr,
    IQ1: tl.constexpr,
    IW0: tl.constexpr,
    KV0: tl.constexpr,
    M: tl.constexpr = 16,
    BN: tl.constexpr = 64,
    KLOOP: tl.constexpr = TMA_PREFILL_KLOOP,
    USE_PDL: tl.constexpr = False,
):
    """Exact long-prefill score using TMA for a compact logical Indexer-K.

    The page table is gathered once outside this kernel.  TMA can then issue
    aligned [BN, 128] key tiles while the query tile is reused across the
    K-loop.  The arithmetic and causal mask match ``_score_tiled_unrolled``.
    """
    first = tl.program_id(0) * M
    row = first + tl.arange(0, M)
    row_offset = row.to(tl.int64) * NK
    pos = tl.load(Pos + row, row < NQ, other=-1)
    qh = tl.arange(0, M * 8)
    qrow = first + qh // 8
    dims = tl.arange(0, 128)
    valid_q = qrow < NQ
    query = tl.load(
        IQ + qrow[:, None] * IQ0 + (qh % 8)[:, None] * IQ1 + dims[None, :],
        valid_q[:, None],
        other=0,
    )
    weights = tl.load(IW + qrow * IW0 + qh % 8, valid_q, other=0).to(tl.float32)
    base = tl.program_id(1) * (BN * KLOOP)
    max_pos = tl.max(pos, 0)
    # The descriptor is created per program so the launch remains shape-generic.
    # KV0 is 128 for the compact row-major gather and satisfies TMA alignment.
    key_desc = tl.make_tensor_descriptor(
        KV,
        shape=[NK, 128],
        strides=[KV0, 1],
        block_shape=[BN, 128],
        padding_option="zero",
    )
    for kt in tl.static_range(0, KLOOP):
        block = base + kt * BN
        col = block + tl.arange(0, BN)
        keep_k = col < NK
        if block <= max_pos:
            keys = key_desc.load([block, 0]).trans(1, 0)
            dot = tl.dot(query, keys, input_precision="ieee")
            score = tl.sum(tl.reshape(tl.maximum(dot, 0.0) * weights[:, None], (M, 8, BN)), 1)
            keep = (row[:, None] < NQ) & keep_k[None, :] & (col[None, :] <= pos[:, None])
            score = tl.where(keep, score, float("-inf"))
        else:
            score = tl.full((M, BN), float("-inf"), tl.float32)
        tl.store(Out + row_offset[:, None] + col[None, :], score, (row[:, None] < NQ) & keep_k[None, :])
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _sort_selected(Keys, Indices, Out, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, B)
    key = tl.load(Keys + row * K + slot, slot < K, other=-9223372036854775808)
    idx = tl.load(Indices + row * K + slot, slot < K, other=0)
    valid = (slot < K) & (key != -9223372036854775808)
    ordered = tl.sort(tl.where(valid, idx.to(tl.int32), 2147483647), descending=False)
    tl.store(Out + row * K + slot, tl.where(ordered == 2147483647, -1, ordered), slot < K)


@triton.jit
def _selected_pages(Indices, Table, Req, Pages, Lengths, K: tl.constexpr, TS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, B)
    request = tl.load(Req + row)
    idx = tl.load(Indices + row * K + slot, slot < K, other=-1)
    valid = (slot < K) & (idx >= 0)
    mem = tl.load(Table + request * TS + idx, valid, other=0)
    tl.store(Pages + row * K + slot, mem, slot < K)
    tl.store(Lengths + row, tl.sum(valid.to(tl.int32), 0))


def _pdl_launch_kwargs(tensor, enabled):
    if not enabled or tensor.device.type != "cuda":
        return {"USE_PDL": False}
    if torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return {"USE_PDL": False}
    return {"USE_PDL": True, "launch_pdl": True}


def ranked_scores(
    iq,
    iw,
    kv,
    table,
    requests,
    positions,
    max_keys,
    attention_dim,
    *,
    packed=True,
    query_tile=4,
    out=None,
    use_pdl=False,
    prefill_unrolled=False,
    contiguous_kv=False,
):
    if max_keys > 1 << INDEX_BITS:
        raise ValueError("Packed DSA selector supports at most 1M key positions")
    if iq.shape[1:] != (8, 128):
        raise ValueError("Tiled selector requires the trained 8x128 Indexer")
    if query_tile not in (4, 8, 16):
        raise ValueError("DSA query tile must be 4, 8, or 16")
    expected_shape = (iq.shape[0], max_keys)
    expected_dtype = torch.int64 if packed else torch.float32
    if out is None:
        output = torch.empty(expected_shape, dtype=expected_dtype, device=iq.device)
    else:
        if out.shape != expected_shape or out.dtype != expected_dtype or out.device != iq.device or not out.is_contiguous():
            raise ValueError("DSA score workspace has an incompatible shape, dtype, or device")
        output = out
    launch_kwargs = dict(
        num_warps=4,
        enable_fp_fusion=False,
        **_pdl_launch_kwargs(iq, use_pdl),
    )
    if prefill_unrolled:
        if packed:
            raise ValueError("prefill_unrolled requires unpacked scores")
        if contiguous_kv:
            if not kv.is_contiguous() or kv.ndim != 2 or kv.shape[1] != 128:
                raise ValueError("contiguous_kv requires a contiguous [tokens, 128] Indexer-K")
            _install_tma_allocator()
        if contiguous_kv and tma_prefill_score_available(iq):
            # TMA handles a compact logical key row.  M=16 is the best H200
            # balance after the gather removes page-table indirection.
            _score_tiled_unrolled_tma[
                (triton.cdiv(iq.shape[0], 16), triton.cdiv(max_keys, 64 * TMA_PREFILL_KLOOP))
            ](
                iq,
                iw,
                kv,
                positions,
                output,
                iq.shape[0],
                max_keys,
                iq.stride(0),
                iq.stride(1),
                iw.stride(0),
                kv.stride(0),
                M=16,
                BN=64,
                KLOOP=TMA_PREFILL_KLOOP,
                num_warps=8,
                **{k: v for k, v in launch_kwargs.items() if k != "num_warps"},
            )
        else:
            _score_tiled_unrolled[
                (triton.cdiv(iq.shape[0], 32), triton.cdiv(max_keys, 64 * 8))
            ](
                iq,
                iw,
                kv,
                table,
                requests,
                positions,
                output,
                iq.shape[0],
                max_keys,
                iq.stride(0),
                iq.stride(1),
                iw.stride(0),
                kv.stride(0),
                table.stride(0),
                M=32,
                BN=64,
                KLOOP=8,
                num_warps=8,
                **{k: v for k, v in launch_kwargs.items() if k != "num_warps"},
            )
    else:
        _score_tiled[(triton.cdiv(iq.shape[0], query_tile), triton.cdiv(max_keys, SCORE_KEY_TILE))](
            iq,
            iw,
            kv,
            table,
            requests,
            positions,
            output,
            iq.shape[0],
            max_keys,
            iq.stride(0),
            iq.stride(1),
            iw.stride(0),
            kv.stride(0),
            table.stride(0),
            attention_dim,
            M=query_tile,
            BN=SCORE_KEY_TILE,
            PACKED=packed,
            **launch_kwargs,
        )
    return output


def packed_topk(keys, topk):
    count = min(topk, keys.shape[1])
    values, indices = keys.topk(count, dim=-1, sorted=False)
    output = torch.empty((keys.shape[0], count), dtype=torch.int32, device=keys.device)
    _sort_selected[(keys.shape[0],)](values, indices, output, count, triton.next_power_of_2(count), num_warps=4)
    return output


def selected_pages(indices, table, requests):
    pages = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    lengths = torch.empty(indices.shape[0], dtype=torch.int32, device=indices.device)
    _selected_pages[(indices.shape[0],)](
        indices,
        table,
        requests,
        pages,
        lengths,
        indices.shape[1],
        table.stride(0),
        triton.next_power_of_2(indices.shape[1]),
        num_warps=4,
    )
    return pages, lengths


def fa3_selected_attention(
    q,
    key_cache,
    value_cache,
    table,
    requests,
    indices,
    flash_func=None,
    *,
    cu_q=None,
    physical_pages=False,
    physical_lengths=None,
):
    if flash_func is None:
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        flash_func = flash_attn_with_kvcache
    if flash_func is None:
        raise RuntimeError("Optimized DSA requires the existing sgl_kernel FA3 backend")
    if physical_pages:
        pages = indices
        lengths = physical_lengths
        if lengths is None:
            lengths = (pages >= 0).sum(dim=-1, dtype=torch.int32)
    else:
        pages, lengths = selected_pages(indices, table, requests)
    nq, _, dim = q.shape
    key = key_cache.unsqueeze(1)
    value = value_cache.unsqueeze(1)
    if cu_q is None:
        cu_q = torch.arange(nq + 1, dtype=torch.int32, device=q.device)
    elif cu_q.shape != (nq + 1,) or cu_q.dtype != torch.int32 or cu_q.device != q.device:
        raise ValueError("DSA FA3 requires one int32 cumulative query entry per row plus the initial zero")
    output = flash_func(
        q=q.contiguous(),
        k_cache=key,
        v_cache=value,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu_q,
        cu_seqlens_k_new=None,
        max_seqlen_q=1,
        causal=False,
        window_size=(-1, -1),
        softmax_scale=dim ** -0.5,
        return_softmax_lse=False,
        num_splits=_fa3_num_splits(nq),
    )
    return torch.where((lengths > 0)[:, None, None], output, 0.0)
