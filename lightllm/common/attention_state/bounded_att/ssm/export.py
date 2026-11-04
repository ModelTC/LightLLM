from math import prod

import torch
import triton
import triton.language as tl

from ..copy_kernel import copy_bounded_att_tensors


def upload_indices(values, device, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


@triton.jit(do_not_specialize=["count"])
def _export_indexed(
    Source,
    Targets,
    Reqs,
    Accepted,
    count,
    SHAPE: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    TARGET_STRIDES: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    total = count.to(tl.int64) * SIZE
    for start in range(tl.program_id(0).to(tl.int64) * BLOCK, total, tl.num_programs(0) * BLOCK):
        row = start + tl.arange(0, BLOCK)
        valid = row < total
        item, offset = row // SIZE, row % SIZE
        req = tl.load(Reqs + item, valid, 0).to(tl.int64)
        accepted = tl.load(Accepted + req, valid, 0).to(tl.int64)
        source_offset = req * REQUEST_STRIDE + accepted * INDEX_STRIDE
        target_offset = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            position = offset % SHAPE[axis]
            offset //= SHAPE[axis]
            source_offset += position * SOURCE_STRIDES[axis]
            target_offset += position * TARGET_STRIDES[axis]
        target = tl.load(Targets + item, valid, 0).to(tl.pointer_type(Source.dtype.element_ty))
        values = tl.load(Source + source_offset, valid, 0)
        tl.store(target + target_offset, values, valid)


def export_indexed_state(source, req_indices, targets, accepted_indices=None, *, slots_per_request=1, index_stride=1):
    if not req_indices:
        return
    assert len(req_indices) == len(targets)
    shape, target_strides = targets[0].shape, targets[0].stride()
    assert all(t.shape == shape and t.stride() == target_strides and t.dtype == source.dtype for t in targets)
    if accepted_indices is None:
        copies = [(source[:, req * slots_per_request][..., : shape[-1]], dst) for req, dst in zip(req_indices, targets)]
        if source.is_cuda:
            copy_bounded_att_tensors(copies)
        else:
            for src, dst in copies:
                dst.copy_(src)
        return
    if not source.is_cuda:
        for req, dst in zip(req_indices, targets):
            accepted = int(accepted_indices[req])
            if slots_per_request > 1:
                dst.copy_(source[:, req * slots_per_request + accepted])
            else:
                dst.copy_(source[:, req, ..., accepted : accepted + shape[-1]])
        return
    _export_indexed[(24,)](
        source,
        upload_indices([t.data_ptr() for t in targets], source.device),
        upload_indices(req_indices, source.device, torch.int32),
        accepted_indices,
        len(req_indices),
        tuple(shape),
        (source.stride(0), *source.stride()[2:]),
        target_strides,
        source.stride(1) * slots_per_request,
        index_stride,
        prod(shape),
        1024,
        num_warps=8,
    )
