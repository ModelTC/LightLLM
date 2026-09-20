"""词表并行 top-k 候选的通信缓冲区打包与解包。

``local_vocab_topk`` 在本地词表分片上返回两个形状为 ``[token_num, candidate_count]``
的 tensor：候选分数和分片内 token ID。本文件用一个 Triton kernel 将分数转为
FP32、token ID 加全局偏移以及两个 tensor 的拼接一次完成，生成下面的通信布局：

    packed[token] = [value_0, ..., value_k-1, id_bits_0, ..., id_bits_k-1]

``all_gather_into_tensor`` 会按 rank 将 ``packed`` 的行连续拼接。第二个 Triton kernel
再把通信结果重排为 ``[token_num, world_size * candidate_count]`` 的分数和全局 token ID。

通信 tensor 使用 FP32。token ID 并非数值转换为浮点数，而是先转为 int32，再将相同的
32 位二进制内容写入 FP32 槽位；解包时执行相反的位转换。这样既能用一个 tensor 完成通信，
也不会受到 FP32 只能连续精确表示到 2^24 的限制。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_vocab_parallel_topk_kernel(
    values,
    token_ids,
    packed,
    values_stride_0,
    values_stride_1,
    token_ids_stride_0,
    token_ids_stride_1,
    packed_stride_0,
    packed_stride_1,
    token_num,
    vocab_start,
    candidate_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 将二维输出 [token_num, candidate_count] 展平，每个 program 处理连续的一段候选。
    element_indexes = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = element_indexes < token_num * candidate_count
    token_indexes = element_indexes // candidate_count
    candidate_indexes = element_indexes % candidate_count

    # 输入为 [token_num, candidate_count]；显式 stride 同时支持连续输出和回退路径的转置视图。
    value_offsets = token_indexes * values_stride_0 + candidate_indexes * values_stride_1
    token_id_offsets = token_indexes * token_ids_stride_0 + candidate_indexes * token_ids_stride_1
    packed_value_offsets = token_indexes * packed_stride_0 + candidate_indexes * packed_stride_1

    # 分数统一转为 FP32；分片内 token ID 加 vocab_start 后变为全局 token ID。
    candidate_values = tl.load(values + value_offsets, mask=valid_mask).to(tl.float32)
    candidate_token_ids = tl.load(token_ids + token_id_offsets, mask=valid_mask) + vocab_start

    # 这里是位转换而非数值转换。例如 int32 的 0x01234567 会以完全相同的 32 位内容
    # 存入 FP32 槽位，通信过程只复制这些位，因此大于 2^24 的 token ID 也不会丢失精度。
    token_id_bits = candidate_token_ids.to(tl.int32).to(tl.float32, bitcast=True)

    # packed 每行前 k 个槽位保存分数，后 k 个槽位保存 token ID 的二进制内容。
    packed_token_id_offsets = packed_value_offsets + candidate_count * packed_stride_1
    tl.store(packed + packed_value_offsets, candidate_values, mask=valid_mask)
    tl.store(packed + packed_token_id_offsets, token_id_bits, mask=valid_mask)


@triton.jit
def _unpack_vocab_parallel_topk_kernel(
    packed,
    values,
    token_ids,
    packed_stride_0,
    packed_stride_1,
    values_stride_0,
    values_stride_1,
    token_ids_stride_0,
    token_ids_stride_1,
    token_num,
    candidate_count: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_width: tl.constexpr = world_size * candidate_count

    # 解包后的每一行依次放置 rank 0 的 k 个候选、rank 1 的 k 个候选，以此类推。
    element_indexes = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = element_indexes < token_num * output_width
    token_indexes = element_indexes // output_width
    output_candidate_indexes = element_indexes % output_width
    source_rank_indexes = output_candidate_indexes // candidate_count
    rank_candidate_indexes = output_candidate_indexes % candidate_count

    # all_gather_into_tensor 沿第 0 维按 rank 拼接：
    # [rank 0 的 token 行][rank 1 的 token 行]...[rank n 的 token 行]。
    packed_row_indexes = source_rank_indexes * token_num + token_indexes
    packed_value_offsets = packed_row_indexes * packed_stride_0 + rank_candidate_indexes * packed_stride_1
    packed_token_id_offsets = packed_value_offsets + candidate_count * packed_stride_1

    output_value_offsets = token_indexes * values_stride_0 + output_candidate_indexes * values_stride_1
    output_token_id_offsets = token_indexes * token_ids_stride_0 + output_candidate_indexes * token_ids_stride_1

    candidate_values = tl.load(packed + packed_value_offsets, mask=valid_mask)
    token_id_bits = tl.load(packed + packed_token_id_offsets, mask=valid_mask)

    # 恢复打包前的 int32 位模式，再扩展为下游索引需要的 int64 类型。
    candidate_token_ids = token_id_bits.to(tl.int32, bitcast=True).to(tl.int64)
    tl.store(values + output_value_offsets, candidate_values, mask=valid_mask)
    tl.store(token_ids + output_token_id_offsets, candidate_token_ids, mask=valid_mask)


@torch.no_grad()
def pack_vocab_parallel_topk(
    values: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_start: int,
    packed: torch.Tensor,
) -> None:
    """将一个 rank 的本地 top-k 结果打包进单个 FP32 通信 tensor。

    参数：
        values: 本地候选分数，形状为 ``[token_num, candidate_count]``，不要求连续。
        token_ids: 对应的分片内 token ID，形状与 ``values`` 相同，类型为 int64。
        vocab_start: 当前 rank 词表分片在完整词表中的起始位置。
        packed: 调用方分配的输出，形状为 ``[token_num, 2 * candidate_count]``，类型为 FP32。

    函数直接写入 ``packed``，不创建额外 tensor。每行前半部分是 FP32 分数，后半部分是
    全局 token ID 的 int32 位模式。
    """

    assert values.is_cuda and token_ids.is_cuda and packed.is_cuda
    assert values.ndim == 2 and values.shape == token_ids.shape
    assert token_ids.dtype == torch.int64 and packed.dtype == torch.float32
    token_num, candidate_count = values.shape
    assert packed.shape == (token_num, candidate_count * 2)
    if packed.numel() == 0:
        return

    element_count = token_num * candidate_count
    block_size = 256
    grid = (triton.cdiv(element_count, block_size),)
    _pack_vocab_parallel_topk_kernel[grid](
        values,
        token_ids,
        packed,
        *values.stride(),
        *token_ids.stride(),
        *packed.stride(),
        token_num=token_num,
        vocab_start=vocab_start,
        candidate_count=candidate_count,
        BLOCK_SIZE=block_size,
    )


@torch.no_grad()
def unpack_vocab_parallel_topk(
    packed: torch.Tensor,
    values: torch.Tensor,
    token_ids: torch.Tensor,
    candidate_count: int,
    world_size: int,
) -> None:
    """将 all-gather 后的通信 tensor 解包为分数和全局 token ID。

    参数：
        packed: all-gather 输出，形状为 ``[world_size * token_num, 2 * candidate_count]``。
        values: 调用方分配的 FP32 分数输出，形状为
            ``[token_num, world_size * candidate_count]``。
        token_ids: 调用方分配的 int64 token ID 输出，形状与 ``values`` 相同。
        candidate_count: 每个 rank 为每个 token 提供的候选数量。
        world_size: 参与词表并行通信的 rank 数量。

    输出的候选维按 rank 分段排列，且每个 rank 内保持本地 ``torch.topk`` 的原始顺序。
    """

    assert packed.is_cuda and values.is_cuda and token_ids.is_cuda
    assert packed.ndim == 2 and values.ndim == 2 and token_ids.ndim == 2
    assert packed.dtype == torch.float32 and values.dtype == torch.float32 and token_ids.dtype == torch.int64
    token_num = values.shape[0]
    output_width = world_size * candidate_count
    assert packed.shape == (world_size * token_num, candidate_count * 2)
    assert values.shape == token_ids.shape == (token_num, output_width)
    if values.numel() == 0:
        return

    element_count = token_num * output_width
    block_size = 256
    grid = (triton.cdiv(element_count, block_size),)
    _unpack_vocab_parallel_topk_kernel[grid](
        packed,
        values,
        token_ids,
        *packed.stride(),
        *values.stride(),
        *token_ids.stride(),
        token_num=token_num,
        candidate_count=candidate_count,
        world_size=world_size,
        BLOCK_SIZE=block_size,
    )
