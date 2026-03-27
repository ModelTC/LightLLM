import torch
import triton
import triton.language as tl

@triton.jit
def update_eagle_mem_indexes_kernel(
    old_indexes_ptr,      # 原始 mem_indexes 指针
    new_indices_ptr,      # 本轮分配的 eagle_mem_indexes_i 指针
    offsets_ptr,          # 每组在 old_indexes 中的起始偏移量 (num_reqs + 1)
    output_ptr,           # 输出的 mem_indexes 指针
    num_reqs,             # 请求数量
    BLOCK_SIZE: tl.constexpr
):
    # 每个 program 处理一个 request 组
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    # 获取当前组的起始位置和长度
    start_idx = tl.load(offsets_ptr + pid)
    next_start_idx = tl.load(offsets_ptr + pid + 1)
    group_size = next_start_idx - start_idx

    # 读取当前组的索引
    # 逻辑：output[0 : size-1] = old[1 : size], output[size-1] = new_val
    for i in range(0, group_size - 1, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < (group_size - 1)
        
        # 移位读取：从 old 的第 1 个位置开始读
        vals = tl.load(old_indexes_ptr + start_idx + offs + 1, mask=mask)
        tl.store(output_ptr + start_idx + offs, vals, mask=mask)

    # 将新的索引填入该组的最后一个位置
    new_val = tl.load(new_indices_ptr + pid)
    tl.store(output_ptr + start_idx + group_size - 1, new_val)

def update_eagle_mem_indexes_triton(old_indexes, new_step_indexes, group_offsets):
    num_reqs = new_step_indexes.shape[0]
    output = torch.empty_like(old_indexes)
    
    # 这里的 BLOCK_SIZE 建议设为 32 或 64，因为 MTP 长度通常不大
    grid = (num_reqs,)
    update_eagle_mem_indexes_kernel[grid](
        old_indexes, new_step_indexes, group_offsets, output,
        num_reqs, BLOCK_SIZE=32
    )
    return output