"""
MTP Diverse Attention - Main Entry Point

MTP (Multi-Token Prediction) Diverse Attention 的实现。

特点：
- 组内请求共享相同的 KV 前缀
- 每个请求 1 个 Q token，但可见 KV 范围不同
- 使用 BLOCK_BATCH 同时处理组内所有请求，一次加载 KV 服务多个 Q

b_mark_shared_group 标记：
- 0: 组内非最后一个请求（跳过）
- N>=1: 一个 N 人组的最后一个请求（需要计算）
  - N=1 表示独立请求

输入 Q 的形状：[batch_size, num_heads, head_dim]
"""
import torch
import triton

from lightllm.utils.infer_utils import calculate_time
from .stage1_single_token import mtp_diverse_stage1_single_token
from .stage2_single_token import mtp_diverse_stage2_single_token


def token_decode_attention_mtp_diverse_single_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    block_seq: int = 256,
    out=None,
    alloc_tensor_func=torch.empty,
):
    """
    MTP Diverse Attention 主入口 - 单 Q token 每请求模式

    在静态 MTP 模式下，每个请求有 1 个 Q token，但第 i 个请求只能看到前 i+1 个 KV。
    使用 BLOCK_BATCH 同时处理组内所有请求，一次加载 KV 服务多个 Q。

    参数：
    - q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    - k: [kv_pool_size, kv_head_num, head_dim] - Key tensor
    - v: [kv_pool_size, kv_head_num, head_dim] - Value tensor
    - Req_to_tokens: [batch, max_kv_len] - 每个 batch 的 KV 索引
    - B_req_idx: [batch] - 每个 batch 的请求索引
    - b_seq_len: [batch] - 每个请求可见的 KV 数量（也是组内位置 +1）
    - b_mark_shared_group: [batch] - 组标记
        - 0: 组内非最后一个请求
        - N>=1: 一个 N 人组的最后一个请求（N=1 表示独立请求）
    - block_seq: 块大小（默认 256）
    - out: 输出 tensor（可选）
    - alloc_tensor_func: tensor 分配函数

    返回：
    - o: [batch_size, num_heads, head_dim] - 输出（和输入 Q 形状相同）
    """
    batch_size = b_seq_len.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # 分配输出 tensor: [batch_size, num_heads, head_dim]
    if out is None:
        o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device)
    else:
        o_tensor = out

    # 从 Req_to_tokens 获取固定的 max_kv_len，保证 CUDA graph 兼容性
    max_kv_len = Req_to_tokens.shape[1]

    # 计算 kv block 数量
    seq_block_num = triton.cdiv(max_kv_len, block_seq)

    # 分配中间结果 buffer
    # mid_o: [batch, head, seq_block_num, head_dim] - 每个 kv block 一个 slot
    mid_o = alloc_tensor_func(
        [batch_size, num_heads, seq_block_num, head_dim],
        dtype=torch.float32,
        device=q.device,
    )
    mid_o_logsumexp = alloc_tensor_func(
        [batch_size, num_heads, seq_block_num],
        dtype=torch.float32,
        device=q.device,
    )

    # Stage1: 计算每个 kv block 的中间结果
    # 使用 BLOCK_BATCH 同时处理组内所有请求，一次加载 KV 服务多个 Q
    mtp_diverse_stage1_single_token(
        q=q,
        k=k,
        v=v,
        Req_to_tokens=Req_to_tokens,
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        max_kv_len=max_kv_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        block_seq=block_seq,
    )

    # Stage2: 将每个请求的中间结果按 seq_len 聚合
    mtp_diverse_stage2_single_token(
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logsumexp,
        B_Seqlen=b_seq_len,
        O=o_tensor,
        block_seq=block_seq,
        max_kv_len=max_kv_len,
    )

    return o_tensor
