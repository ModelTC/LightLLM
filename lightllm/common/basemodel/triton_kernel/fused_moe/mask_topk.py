import torch
import triton
import triton.language as tl


@triton.jit
def _mask_topk(Weights, Ids, Mask, OutWeights, OutIds, ROWS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * K
    active = tl.load(Mask + index // K, valid, other=False)
    weights = tl.load(Weights + index, valid, other=0)
    ids = tl.load(Ids + index, valid, other=-1).to(OutIds.dtype.element_ty)
    tl.store(OutWeights + index, tl.where(active, weights, 0), valid)
    tl.store(OutIds + index, tl.where(active, ids, -1), valid)


def mask_topk(weights, ids, mask, index_dtype=None):
    index_dtype = ids.dtype if index_dtype is None else index_dtype
    if not weights.is_cuda or not weights.is_contiguous() or not ids.is_contiguous() or not mask.is_contiguous():
        return weights.masked_fill(~mask[:, None], 0), ids.masked_fill(~mask[:, None], -1).to(index_dtype)
    output_weights = torch.empty_like(weights)
    output_ids = torch.empty_like(ids, dtype=index_dtype)
    if ids.numel():
        _mask_topk[(triton.cdiv(ids.numel(), 128),)](
            weights, ids, mask, output_weights, output_ids, ids.shape[0], ids.shape[1], 128, num_warps=4
        )
    return output_weights, output_ids
