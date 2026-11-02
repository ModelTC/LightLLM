from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ...triton_kernel.repack_kv_index import repack_kv_index


def page_index_memory_demands(backend):
    size = backend.model.graph_max_batch_size * backend.max_page_num
    if not size:
        return ()
    return backend.workspace_memory_demands(
        f"flashinfer_page_indices_{backend.model.att_state_role}_{backend.infer_page_size}",
        size,
        lanes=2,
        dtype=torch.int32,
    )


def reserve_page_index_buffers(backend):
    size = backend.model.graph_max_batch_size * backend.max_page_num
    backend.kv_indices_buffer = (
        [
            backend.get_gpu_workspace_buffer(
                f"flashinfer_page_indices_{backend.model.att_state_role}_{backend.infer_page_size}",
                size,
                dtype=torch.int32,
                lane_id=lane,
            )
            for lane in range(2)
        ]
        if size
        else []
    )


@triton.jit
def _page_indptr(lengths, starts, last, length_stride, N: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    length = tl.load(lengths + rows * length_stride, rows < N, other=0).to(tl.int32)
    pages = tl.cdiv(length, PAGE)
    ends = tl.cumsum(pages)
    tl.store(starts + rows + 1, ends, rows < N)
    tl.store(starts, 0)
    tl.store(last + rows, tl.where(length > 0, length - (pages - 1) * PAGE, 0), rows < N)


@dataclass(frozen=True)
class PageMetadata:
    starts: torch.Tensor
    last_page_len: torch.Tensor
    indices: torch.Tensor


def build_page_metadata(infer, page_size, indices):
    batch = infer.b_req_idx.numel()
    starts = torch.empty(batch + 1, dtype=torch.int32, device=indices.device)
    last = torch.empty(batch, dtype=torch.int32, device=indices.device)
    _page_indptr[(1,)](
        infer.b_seq_len,
        starts,
        last,
        infer.b_seq_len.stride(0),
        batch,
        page_size,
        triton.next_power_of_2(max(batch, 1)),
    )
    repack_kv_index(
        infer.req_manager.req_to_token_indexs,
        infer.b_req_idx,
        infer.b_seq_len,
        starts[:-1],
        infer.max_kv_seq_len,
        indices,
        page_size,
    )
    return PageMetadata(starts, last, indices)


def prepare_page_metadata(backend, infer):

    key = backend.infer_page_size, backend.max_page_num, infer.is_prefill, infer.microbatch_index
    shared = getattr(infer, "_paged_attention_metadata", None)
    if shared is None:
        shared = infer._paged_attention_metadata = {}
    if key not in shared:
        model = backend.model
        batch_page_num = triton.cdiv(infer.max_kv_seq_len, backend.infer_page_size)
        if (
            not infer.is_prefill
            and infer.batch_size <= model.graph_max_batch_size
            and infer.max_kv_seq_len <= model.graph_max_len_in_batch
            and batch_page_num <= backend.max_page_num
        ):
            indices = backend.kv_indices_buffer[infer.microbatch_index][: infer.batch_size * backend.max_page_num]
        else:
            indices = torch.empty(
                infer.batch_size * max(batch_page_num, backend.max_page_num),
                dtype=torch.int32,
                device=infer.input_ids.device,
            )
        shared[key] = build_page_metadata(infer, backend.infer_page_size, indices)
    return shared[key]
