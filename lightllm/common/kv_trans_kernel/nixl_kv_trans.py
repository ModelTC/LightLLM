import torch

import triton
import triton.language as tl


@triton.jit
def _page_io(
    mem_index_ptr,
    k_page_ptr,
    k_page_stride_size,
    k_page_stride_layer_num,
    k_page_stride_head,
    k_page_stride_dim,
    v_page_ptr,
    v_page_stride_size,
    v_page_stride_layer_num,
    v_page_stride_head,
    v_page_stride_dim,
    k_ptr,
    k_stride_layer_num,
    k_stride_size,
    k_stride_head,
    k_stride_dim,
    v_ptr,
    v_stride_layer_num,
    v_stride_size,
    v_stride_head,
    v_stride_dim,
    page_head_start,
    layer_num,
    head_dim,
    HEAD_DIM_BLOCK: tl.constexpr,
    IS_WRITE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    k_page_stride_size = tl.cast(k_page_stride_size, dtype=tl.int64)
    v_page_stride_size = tl.cast(v_page_stride_size, dtype=tl.int64)

    k_stride_layer_num = tl.cast(k_stride_layer_num, dtype=tl.int64)
    v_stride_layer_num = tl.cast(v_stride_layer_num, dtype=tl.int64)
    k_stride_size = tl.cast(k_stride_size, dtype=tl.int64)
    v_stride_size = tl.cast(v_stride_size, dtype=tl.int64)
    
    tid = tl.program_id(0)
    kv_head_id = tl.program_id(1)
    page_head_id = page_head_start + kv_head_id

    mem_index = tl.load(mem_index_ptr + tid)
    off_dim = tl.arange(0, HEAD_DIM_BLOCK)
    if NEED_MASK:
        mask = off_dim < head_dim
    else:
        mask = None

    for layer_index in tl.range(layer_num, num_stages=3):
        if IS_WRITE:
            k_tensor = tl.load(k_ptr + layer_index * k_stride_layer_num + mem_index * k_stride_size + kv_head_id * k_stride_head + off_dim * k_stride_dim, mask=mask)
            v_tensor = tl.load(v_ptr + layer_index * v_stride_layer_num + mem_index * v_stride_size + kv_head_id * v_stride_head + off_dim * v_stride_dim, mask=mask)
            tl.store(k_page_ptr + tid * k_page_stride_size + layer_index * k_page_stride_layer_num + page_head_id * k_page_stride_head + off_dim * k_page_stride_dim, k_tensor, mask=mask)
            tl.store(v_page_ptr + tid * v_page_stride_size + layer_index * v_page_stride_layer_num + page_head_id * v_page_stride_head + off_dim * v_page_stride_dim, v_tensor, mask=mask)
        else:
            k_page_tensor = tl.load(k_page_ptr + tid * k_page_stride_size + layer_index * k_page_stride_layer_num + page_head_id * k_page_stride_head + off_dim * k_page_stride_dim, mask=mask)
            v_page_tensor = tl.load(v_page_ptr + tid * v_page_stride_size + layer_index * v_page_stride_layer_num + page_head_id * v_page_stride_head + off_dim * v_page_stride_dim, mask=mask)
            tl.store(k_ptr + layer_index * k_stride_layer_num + mem_index * k_stride_size + kv_head_id * k_stride_head + off_dim * k_stride_dim, k_page_tensor, mask=mask)
            tl.store(v_ptr + layer_index * v_stride_layer_num + mem_index * v_stride_size + kv_head_id * v_stride_head + off_dim * v_stride_dim, v_page_tensor, mask=mask)
    return

def page_io(mem_indexes:torch.Tensor, page_tensor: torch.Tensor, kv_buffer: torch.Tensor, tp_index:int, tp_world_size:int, mode:str):
    assert mode in ["read", "write"]
    assert mem_indexes.is_contiguous()
    assert page_tensor.is_contiguous()
    assert kv_buffer.is_contiguous()

    page_size, layer_num, page_head_num, page_head_dim = page_tensor.shape
    _layer_num, size, kv_head_num, head_dim = kv_buffer.shape
    repeat_count = (kv_head_num * tp_world_size) // page_head_num
    assert layer_num == _layer_num
    assert len(mem_indexes) <= page_size
    assert page_head_dim == head_dim

    page_k_head_num, page_v_head_num = page_head_num // 2, page_head_num // 2
    k_page_tensor = page_tensor[:, :, 0:page_k_head_num, :]
    v_page_tensor = page_tensor[:, : -page_v_head_num:, :]

    k_head_num, v_head_num = kv_head_num // 2, kv_head_num // 2
    k_buffer = kv_buffer[:, :, 0:k_head_num, :]
    v_buffer = kv_buffer[:, :, -v_head_num:, :]
    
    tp_index = tp_index // repeat_count
    tp_world_size = tp_world_size // repeat_count

    page_write_head_num = page_k_head_num // tp_world_size
    page_head_start = tp_index * (page_write_head_num)

    token_num = len(mem_indexes)
    grid = (token_num, page_write_head_num)

    _page_io[grid](
        mem_index_ptr=mem_indexes,
        k_page_ptr=k_page_tensor,
        k_page_stride_size=k_page_tensor.stride(0),
        k_page_stride_layer_num=k_page_tensor.stride(1),
        k_page_stride_head=k_page_tensor.stride(2),
        k_page_stride_dim=k_page_tensor.stride(3),
        v_page_ptr=v_page_tensor,
        v_page_stride_size=v_page_tensor.stride(0),
        v_page_stride_layer_num=v_page_tensor.stride(1),
        v_page_stride_head=v_page_tensor.stride(2),
        v_page_stride_dim=v_page_tensor.stride(3),
        k_ptr=k_buffer,
        k_stride_layer_num=k_buffer.stride(0),
        k_stride_size=k_buffer.stride(1),
        k_stride_head=k_buffer.stride(2),
        k_stride_dim=k_buffer.stride(3),
        v_ptr=v_buffer,
        v_stride_layer_num=v_buffer.stride(0),
        v_stride_size=v_buffer.stride(1),
        v_stride_head=v_buffer.stride(2),
        v_stride_dim=v_buffer.stride(3),
        page_head_start=page_head_start,
        layer_num=layer_num,
        head_dim=head_dim,
        HEAD_DIM_BLOCK=triton.next_power_of_2(head_dim),
        IS_WRITE=mode=="write",
        NEED_MASK=triton.next_power_of_2(head_dim) != head_dim,
        num_warps=1,
    )
    return
