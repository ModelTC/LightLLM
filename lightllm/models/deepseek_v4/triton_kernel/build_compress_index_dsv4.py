import torch
import triton
import triton.language as tl


@triton.jit
def _build_compress_index_kernel(
    req_idx_ptr,
    pos_ptr,
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_c_ptr,
    index_ptr,
    length_ptr,
    cap,
    RATIO: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0)
    eb = tl.program_id(1)
    req = tl.load(req_idx_ptr + t).to(tl.int64)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    raw_len = (pos + 1) // RATIO

    e = eb * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e < cap
    valid = (e < raw_len) & e_mask
    # group-end token of compressed entry e: position e*RATIO + (RATIO-1).
    end_pos = e * RATIO + (RATIO - 1)
    safe_pos = tl.where(valid, end_pos, 0)
    full_slot = tl.load(req_to_token_ptr + req * req_to_token_stride0 + safe_pos, mask=valid, other=0).to(tl.int64)
    c_slot = tl.load(full_to_c_ptr + full_slot, mask=valid, other=-1).to(tl.int32)
    tl.store(index_ptr + t * cap + e, c_slot, mask=e_mask)

    if eb == 0:
        tl.store(length_ptr + t, tl.maximum(raw_len, 1).to(tl.int32))


def build_compress_index(
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    full_to_c_indexs: torch.Tensor,
    ratio: int,
    cap: int,
):
    """Fused two-level group-end gather for the c4/c128 compressed-entry index tables.

    For token t (at request `req_idx[t]`, absolute `positions[t]`) and compressed entry e:
        slot[t, e] = full_to_c[ req_to_token[req, e*ratio + (ratio-1)] ]   (the group-end token's full slot)
    with slot = -1 where e >= (pos+1)//ratio (beyond the causal compressed length) or where the
    full->c map is unset. Returns (index [T, cap] int32, length [T] int32 = clamp((pos+1)//ratio, 1)).

    Replaces the eager _gather_compress_slots/_c128/c4-causal torch chain. `cap` must be a multiple of
    64 (FlashMLA topk alignment); the tiled grid (T, ceil(cap/BLOCK_E)) scales to 1M-context caps.
    cuda-graph-safe: cap is fixed per graph bucket, shapes static.
    """
    T = positions.shape[0]
    index = torch.empty((T, cap), dtype=torch.int32, device=positions.device)
    length = torch.empty((T,), dtype=torch.int32, device=positions.device)
    if T == 0:
        return index, length
    BLOCK_E = 256
    grid = (T, triton.cdiv(cap, BLOCK_E))
    _build_compress_index_kernel[grid](
        req_idx,
        positions,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        full_to_c_indexs,
        index,
        length,
        cap,
        RATIO=ratio,
        BLOCK_E=BLOCK_E,
        num_warps=4,
    )
    return index, length
