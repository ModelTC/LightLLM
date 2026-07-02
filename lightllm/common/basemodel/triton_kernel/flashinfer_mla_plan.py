import torch

import triton
import triton.language as tl


# 这个文件不实现 MLA attention 本身，只在 decode CUDA graph replay 时重新生成
# FlashInfer MLA 的 plan 表。FlashInfer run kernel 会从
# decode_wrapper._int_workspace_buffer 中读取这些表：
#
#   q_indptr / kv_indptr / q_len / kv_len / q_start / kv_start / kv_end:
#       attention 阶段消费的 work item 描述。
#   partial_indptr:
#       -1 表示直接写最终输出；否则写到 partial 输出。
#   merge_*:
#       kernel 内 split-K merge 阶段消费的描述。
#   work_indptr:
#       每个 cluster 在 work-item 表中的范围。
#
# 这些数组的 offset 由 FlashInfer 首次 CPU plan 生成，并保存在
# decode_wrapper._plan_info 中。这里保持相同的 buffer layout，只用 Triton 覆写
# 数组内容。


@triton.jit
def _fill_exact_mla_decode_plan_kernel(
    int_buf_i32,
    kv_indptr,
    q_indptr_off: tl.constexpr,
    kv_indptr_off: tl.constexpr,
    partial_indptr_off: tl.constexpr,
    merge_start_off: tl.constexpr,
    merge_end_off: tl.constexpr,
    merge_partial_start_off: tl.constexpr,
    merge_partial_end_off: tl.constexpr,
    merge_stride_off: tl.constexpr,
    q_len_off: tl.constexpr,
    kv_len_off: tl.constexpr,
    q_start_off: tl.constexpr,
    kv_start_off: tl.constexpr,
    kv_end_off: tl.constexpr,
    work_indptr_off: tl.constexpr,
    batch_size: tl.constexpr,
    num_clusters: tl.constexpr,
    total_ctas: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # exact non-split 路径：每个 request 只有一个 work item。它直接使用真实
    # kv_indptr，也不会写 partial 输出，是最简单、最稳的 CUDA graph replay plan。
    cluster_offsets = tl.arange(0, BLOCK_C)
    base_count = batch_size // num_clusters
    extra_count = batch_size - base_count * num_clusters
    work_indptr = cluster_offsets * base_count + tl.minimum(cluster_offsets, extra_count)
    tl.store(
        int_buf_i32 + work_indptr_off + cluster_offsets,
        work_indptr,
        mask=cluster_offsets <= num_clusters,
    )

    # 这条路径没有 work 会写 partial 输出，所以需要把所有 merge CTA 的 range
    # 都置零，相当于禁用 merge 阶段。
    zeros = tl.full((BLOCK_C,), 0, tl.int32)
    cta_offsets = tl.arange(0, BLOCK_C)
    tl.store(int_buf_i32 + merge_start_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)
    tl.store(int_buf_i32 + merge_end_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)
    tl.store(
        int_buf_i32 + merge_partial_start_off + cta_offsets,
        zeros,
        mask=cta_offsets < total_ctas,
    )
    tl.store(
        int_buf_i32 + merge_partial_end_off + cta_offsets,
        zeros,
        mask=cta_offsets < total_ctas,
    )
    tl.store(int_buf_i32 + merge_stride_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)

    batch_offsets = tl.arange(0, BLOCK_B)
    valid_batch = batch_offsets < batch_size
    cluster = batch_offsets % num_clusters
    rank_in_cluster = batch_offsets // num_clusters
    # 按 cluster 维度拉平 work item，这样 work_indptr[cluster] 可以指向一段
    # 连续 work range，和 FlashInfer scheduler 的约定保持一致。
    record_index = cluster * base_count + tl.minimum(cluster, extra_count) + rank_in_cluster

    kv_start = tl.load(kv_indptr + batch_offsets, mask=valid_batch, other=0)
    kv_next = tl.load(kv_indptr + batch_offsets + 1, mask=valid_batch, other=0)
    kv_len = kv_next - kv_start

    tl.store(int_buf_i32 + q_indptr_off + record_index, batch_offsets, mask=valid_batch)
    tl.store(int_buf_i32 + kv_indptr_off + record_index, kv_start, mask=valid_batch)
    tl.store(int_buf_i32 + partial_indptr_off + record_index, -1, mask=valid_batch)
    tl.store(int_buf_i32 + q_len_off + record_index, 1, mask=valid_batch)
    tl.store(int_buf_i32 + kv_len_off + record_index, kv_len, mask=valid_batch)
    tl.store(int_buf_i32 + q_start_off + record_index, 0, mask=valid_batch)
    tl.store(int_buf_i32 + kv_start_off + record_index, 0, mask=valid_batch)
    tl.store(int_buf_i32 + kv_end_off + record_index, kv_len, mask=valid_batch)


@triton.jit
def _fill_fixed_chunk_mla_decode_plan_kernel(
    int_buf_i32,
    kv_indptr,
    q_indptr_off: tl.constexpr,
    kv_indptr_off: tl.constexpr,
    partial_indptr_off: tl.constexpr,
    merge_start_off: tl.constexpr,
    merge_end_off: tl.constexpr,
    merge_partial_start_off: tl.constexpr,
    merge_partial_end_off: tl.constexpr,
    merge_stride_off: tl.constexpr,
    q_len_off: tl.constexpr,
    kv_len_off: tl.constexpr,
    q_start_off: tl.constexpr,
    kv_start_off: tl.constexpr,
    kv_end_off: tl.constexpr,
    work_indptr_off: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    cluster_size: tl.constexpr,
    num_clusters: tl.constexpr,
    total_ctas: tl.constexpr,
    min_chunk_size: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # fixed-chunk split 路径，仅面向 decode q_len=1。只要
    # num_heads <= cluster_size * 64，每个 request 就只有一个 Q tile。长 KV
    # 会被拆成多个 work item，再由 FlashInfer 已有的 kernel 内 persistent merge
    # 合并。
    batch_offsets = tl.arange(0, BLOCK_B)
    valid_batch = batch_offsets < batch_size

    kv_start = tl.load(kv_indptr + batch_offsets, mask=valid_batch, other=0)
    kv_next = tl.load(kv_indptr + batch_offsets + 1, mask=valid_batch, other=0)
    kv_len = kv_next - kv_start

    # 使用 GPU 上真实 kv_indptr 来模拟 FlashInfer CPU scheduler：
    #   kv_len_limit = f(total_kv_len / num_clusters)
    # 这是它和 conservative max-kv plan 的关键区别：不依赖 host meta，也能适配
    # 长短混合 batch。
    total_kv_len = tl.load(kv_indptr + batch_size) - tl.load(kv_indptr)
    avg_kv_len = (total_kv_len + num_clusters - 1) // num_clusters
    chunk_hint = tl.where(
        avg_kv_len <= 8,
        32,
        tl.where(
            avg_kv_len <= 16,
            64,
            tl.where(avg_kv_len <= 32, 128, tl.where(avg_kv_len <= 64, 192, ((avg_kv_len + 255) // 256) * 256)),
        ),
    )
    chunk_size = tl.maximum(chunk_hint, min_chunk_size)
    chunks = (kv_len + chunk_size - 1) // chunk_size
    chunks = tl.where(valid_batch, tl.maximum(chunks, 1), 0)

    # work item 按 request-major 排列：request 0 的所有 chunk 在前，然后是
    # request 1，以此类推。下面的 work_indptr 会把这个扁平 work range 均匀
    # 切给 FlashInfer 的各个 cluster。
    chunk_prefix = tl.cumsum(chunks, 0) - chunks
    total_chunks = tl.sum(chunks, 0)

    # partial_indptr 的单位是 packed Q row。decode q_len=1 时 row tile size
    # 就是 num_heads；split request 的每个 KV chunk 都需要在 partial output
    # workspace 中占用一个 row_tile_size 切片。
    row_tile_size: tl.constexpr = num_heads
    partial_rows = tl.where(chunks > 1, chunks * row_tile_size, 0)
    partial_base = tl.cumsum(partial_rows, 0) - partial_rows

    cluster_offsets = tl.arange(0, BLOCK_C)
    work_start = (total_chunks * cluster_offsets) // num_clusters
    tl.store(
        int_buf_i32 + work_indptr_off + cluster_offsets,
        work_start,
        mask=cluster_offsets <= num_clusters,
    )

    # 先清空所有 merge CTA。只有 split request 会在下面写入有效 merge 描述。
    # FlashInfer 会把 zero-length merge range 当成 no-op。
    zeros = tl.full((BLOCK_C,), 0, tl.int32)
    cta_offsets = tl.arange(0, BLOCK_C)
    tl.store(int_buf_i32 + merge_start_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)
    tl.store(int_buf_i32 + merge_end_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)
    tl.store(
        int_buf_i32 + merge_partial_start_off + cta_offsets,
        zeros,
        mask=cta_offsets < total_ctas,
    )
    tl.store(
        int_buf_i32 + merge_partial_end_off + cta_offsets,
        zeros,
        mask=cta_offsets < total_ctas,
    )
    tl.store(int_buf_i32 + merge_stride_off + cta_offsets, zeros, mask=cta_offsets < total_ctas)

    is_split = valid_batch & (chunks > 1)
    split_count = tl.sum(is_split.to(tl.int32), 0)
    # FlashInfer 启动 total_ctas = num_blks_x * num_blks_y 个 CTA。merge 表
    # 每个 CTA 只有一个 entry，所以 split request 需要共享这份容量。如果只有
    # 一个 request 很长，它可以使用较多 merge CTA；如果很多 request 都很长，
    # 每个 request 分到的 row chunk 就会减少。
    merge_capacity = total_ctas // tl.maximum(split_count, 1)
    # merge work 沿 packed row/head 方向切分。merge_chunks 越大，表示越多
    # CTA 并行 merge 不同 head range。
    merge_chunks = tl.minimum(num_heads, tl.minimum(chunks * cluster_size, merge_capacity))
    merge_chunks_for_div = tl.maximum(merge_chunks, 1)
    row_chunk_size = (num_heads + merge_chunks_for_div - 1) // merge_chunks_for_div
    merge_chunks = (num_heads + row_chunk_size - 1) // row_chunk_size
    merge_chunks = tl.where(is_split, merge_chunks, 0)
    merge_base = tl.cumsum(merge_chunks, 0) - merge_chunks
    merge_offsets = tl.arange(0, BLOCK_M)
    valid_merge = merge_offsets[None, :] < merge_chunks[:, None]
    local_merge_start = merge_offsets[None, :] * row_chunk_size[:, None]
    local_merge_end = tl.minimum(local_merge_start + row_chunk_size[:, None], row_tile_size)
    merge_index = merge_base[:, None] + merge_offsets[None, :]
    tl.store(
        int_buf_i32 + merge_start_off + merge_index,
        batch_offsets[:, None] * num_heads + local_merge_start,
        mask=valid_merge,
    )
    tl.store(
        int_buf_i32 + merge_end_off + merge_index,
        batch_offsets[:, None] * num_heads + local_merge_end,
        mask=valid_merge,
    )
    tl.store(
        int_buf_i32 + merge_partial_start_off + merge_index,
        partial_base[:, None] + local_merge_start,
        mask=valid_merge,
    )
    tl.store(
        int_buf_i32 + merge_partial_end_off + merge_index,
        partial_base[:, None] + chunks[:, None] * row_tile_size,
        mask=valid_merge,
    )
    tl.store(int_buf_i32 + merge_stride_off + merge_index, row_tile_size, mask=valid_merge)

    # 写入真正的 attention work-item 表。非 split request 保持
    # partial_indptr=-1，直接写最终输出。split request 把每个 chunk 的 partial
    # 写到：
    #   partial_base + chunk_idx * row_tile_size
    # 上面的 merge 表随后会归并这些 partial row。
    chunk_offsets = tl.arange(0, BLOCK_K)
    work_id = chunk_prefix[:, None] + chunk_offsets[None, :]
    valid_chunk = valid_batch[:, None] & (chunk_offsets[None, :] < chunks[:, None])
    chunk_start = chunk_offsets[None, :] * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, kv_len[:, None])
    partial_indptr = tl.where(
        chunks[:, None] > 1,
        partial_base[:, None] + chunk_offsets[None, :] * row_tile_size,
        -1,
    )

    tl.store(int_buf_i32 + q_indptr_off + work_id, batch_offsets[:, None], mask=valid_chunk)
    tl.store(int_buf_i32 + kv_indptr_off + work_id, kv_start[:, None], mask=valid_chunk)
    tl.store(int_buf_i32 + partial_indptr_off + work_id, partial_indptr, mask=valid_chunk)
    tl.store(int_buf_i32 + q_len_off + work_id, 1, mask=valid_chunk)
    tl.store(int_buf_i32 + kv_len_off + work_id, kv_len[:, None], mask=valid_chunk)
    tl.store(int_buf_i32 + q_start_off + work_id, 0, mask=valid_chunk)
    tl.store(int_buf_i32 + kv_start_off + work_id, chunk_start, mask=valid_chunk)
    tl.store(int_buf_i32 + kv_end_off + work_id, chunk_end, mask=valid_chunk)


@torch.no_grad()
def fill_exact_mla_decode_plan(decode_wrapper, kv_indptr: torch.Tensor, batch_size: int) -> None:
    plan_info = [int(v) for v in decode_wrapper._plan_info]
    int_buf_i32 = decode_wrapper._int_workspace_buffer.view(torch.int32)
    # FlashInfer plan offset 是 byte offset；Triton 这里按 int32 元素写入。
    offsets = [v // 4 for v in plan_info]
    # plan_info[0] 是 grid.x：一个 cluster 内的 CTA 数。
    # plan_info[1] 是 grid.y：cluster 数。
    num_blks_x = plan_info[0]
    num_blks_y = plan_info[1]
    block_b = triton.next_power_of_2(max(batch_size, 1))
    block_c = triton.next_power_of_2(max(num_blks_x * num_blks_y, num_blks_y + 1, 1))

    _fill_exact_mla_decode_plan_kernel[(1,)](
        int_buf_i32=int_buf_i32,
        kv_indptr=kv_indptr,
        q_indptr_off=offsets[2],
        kv_indptr_off=offsets[3],
        partial_indptr_off=offsets[4],
        merge_start_off=offsets[5],
        merge_end_off=offsets[6],
        merge_partial_start_off=offsets[7],
        merge_partial_end_off=offsets[8],
        merge_stride_off=offsets[9],
        q_len_off=offsets[10],
        kv_len_off=offsets[11],
        q_start_off=offsets[12],
        kv_start_off=offsets[13],
        kv_end_off=offsets[14],
        work_indptr_off=offsets[15],
        batch_size=batch_size,
        num_clusters=num_blks_y,
        total_ctas=num_blks_x * num_blks_y,
        BLOCK_B=block_b,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return


def _mla_kv_len_limit_hint(avg_kv_len: int) -> int:
    # 保持和 FlashInfer MLAPlan scheduler 相同的分段函数。这个值对齐后，
    # 生成的 split plan 在长上下文性能上才会接近原生 CPU plan。
    if avg_kv_len <= 8:
        return 32
    if avg_kv_len <= 16:
        return 64
    if avg_kv_len <= 32:
        return 128
    if avg_kv_len <= 64:
        return 192
    return triton.cdiv(avg_kv_len, 256) * 256


@torch.no_grad()
def fill_fixed_chunk_mla_decode_plan(
    decode_wrapper,
    kv_indptr: torch.Tensor,
    batch_size: int,
    num_heads: int,
    max_kv_len: int,
) -> bool:
    plan_info = [int(v) for v in decode_wrapper._plan_info]
    # FlashInfer 启动 MLA run kernel 时使用 grid=(num_blks_x, num_blks_y)。
    # num_blks_x 同时也是 scheduler 中的 cluster_size。每个 CTA 覆盖 64 个
    # packed Q row，所以一个 cluster 覆盖 cluster_tile_q 个 row。
    num_blks_x = plan_info[0]
    num_blks_y = plan_info[1]
    num_ctas = num_blks_x * num_blks_y
    cluster_tile_q = num_blks_x * 64
    row_tile_size = num_heads

    # 计入 plan 生成开销后，极短 decode 使用 exact non-split 更快。当前 split
    # kernel 也有意限制在 q_len=1 且每个 request 只有一个 Q tile 的场景。
    if batch_size <= 0 or max_kv_len <= 512 or row_tile_size <= 0:
        return False
    if num_heads > cluster_tile_q or batch_size > num_ctas:
        return False

    # 这个上界用于确定 BLOCK_K，同时保护 FlashInfer 固定的
    # max_total_num_works=16384 表容量。实际 chunk size 仍会在 Triton kernel
    # 内根据真实 GPU kv_indptr 重新计算。
    min_chunk_size = _mla_kv_len_limit_hint(triton.cdiv(max_kv_len, num_blks_y))
    if min_chunk_size >= max_kv_len:
        return False

    # FlashInfer MLAPlan 里 work-item 相关数组固定按 max_total_num_works=16384
    # 分配。这里用 graph shape 的最坏情况估算每个 request 最多会被拆成多少
    # chunk，如果 batch_size * max_chunks_per_request 超过 16384，就不能安全
    # 覆写这个 plan layout，必须回退 exact non-split。
    max_chunks_per_request = triton.cdiv(max_kv_len, min_chunk_size)
    if batch_size * max_chunks_per_request > 16384:
        return False

    int_buf_i32 = decode_wrapper._int_workspace_buffer.view(torch.int32)
    # FlashInfer plan offset 是 byte offset；Triton 这里按 int32 元素写入。
    offsets = [v // 4 for v in plan_info]
    block_b = triton.next_power_of_2(max(batch_size, 1))
    block_k = triton.next_power_of_2(max(max_chunks_per_request, 1))
    block_c = triton.next_power_of_2(max(num_ctas, num_blks_y + 1, 1))

    _fill_fixed_chunk_mla_decode_plan_kernel[(1,)](
        int_buf_i32=int_buf_i32,
        kv_indptr=kv_indptr,
        q_indptr_off=offsets[2],
        kv_indptr_off=offsets[3],
        partial_indptr_off=offsets[4],
        merge_start_off=offsets[5],
        merge_end_off=offsets[6],
        merge_partial_start_off=offsets[7],
        merge_partial_end_off=offsets[8],
        merge_stride_off=offsets[9],
        q_len_off=offsets[10],
        kv_len_off=offsets[11],
        q_start_off=offsets[12],
        kv_start_off=offsets[13],
        kv_end_off=offsets[14],
        work_indptr_off=offsets[15],
        batch_size=batch_size,
        num_heads=num_heads,
        cluster_size=num_blks_x,
        num_clusters=num_blks_y,
        total_ctas=num_ctas,
        min_chunk_size=min_chunk_size,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        BLOCK_C=block_c,
        BLOCK_M=triton.next_power_of_2(max(num_heads, 1)),
        num_warps=8,
    )
    return True


@torch.no_grad()
def fill_mla_decode_plan_for_cuda_graph(
    decode_wrapper,
    kv_indptr: torch.Tensor,
    batch_size: int,
    num_heads: int,
    max_kv_len: int,
) -> str:
    # 长 decode 优先使用 split plan，因为它能匹配 FlashInfer 的 split-K 并行度。
    # 短序列或当前不支持的 graph shape 回退到 exact plan 来保证正确性。
    use_fixed_chunk_split = fill_fixed_chunk_mla_decode_plan(
        decode_wrapper,
        kv_indptr,
        batch_size,
        num_heads,
        max_kv_len,
    )
    if use_fixed_chunk_split:
        return "fixed_chunk_split"

    fill_exact_mla_decode_plan(decode_wrapper, kv_indptr, batch_size)
    return "exact_non_split"
