# 为 diverse mode 定制设计的 int8kv flash decoding attention 实现，可以实现更高效的多样性采样
import torch

from lightllm.common.attention.api.batch import AttentionBatch
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size

from .int8kv_flash_decoding_diverse_stage1 import flash_decode_stage1
from .int8kv_flash_decoding_diverse_stage2 import flash_decode_stage2
from .int8kv_flash_decoding_diverse_stage3 import flash_diverse_decode_stage3


def token_decode_attention_flash_decoding(
    q,
    batch: AttentionBatch,
    cache_k,
    cache_k_scale,
    cache_v,
    cache_v_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    b_shared_seq_len=None,
    b_mark_shared_group=None,
    shared_streams_dict=None,
):
    if b_shared_seq_len is None or b_mark_shared_group is None or shared_streams_dict is None:
        raise ValueError("diverse attention requires prepared group metadata and slot-owned streams")

    if "stream1" not in shared_streams_dict:
        shared_streams_dict["stream1"] = torch.cuda.Stream()
    if "stream2" not in shared_streams_dict:
        shared_streams_dict["stream2"] = torch.cuda.Stream()

    stream1 = shared_streams_dict["stream1"]
    stream2 = shared_streams_dict["stream2"]

    q_head_num = q.shape[1]
    head_dim = q.shape[2]

    BLOCK_SEQ = 256
    batch_size = batch.batch_size
    max_kv_seq_len = batch.max_kv_seq_len
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 2, head_dim], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 2], dtype=torch.float32, device="cuda"
    )

    current_stream = torch.cuda.current_stream()

    stream1.wait_stream(current_stream)
    with torch.cuda.stream(stream1):
        flash_decode_stage1(
            q=q.view(calcu_shape1),
            k=cache_k,
            k_scale=cache_k_scale,
            v=cache_v,
            v_scale=cache_v_scale,
            Req_to_tokens=batch.req_to_token_indexs,
            B_req_idx=batch.b_req_idx,
            b_shared_seq_len=b_shared_seq_len,
            b_mark_shared_group=b_mark_shared_group,
            max_len_in_batch=batch.max_kv_seq_len,
            mid_out=mid_o,
            mid_out_logsumexp=mid_o_logexpsum,
            block_seq=BLOCK_SEQ,
            max_batch_group_size=get_diverse_max_batch_shared_group_size(),
        )
    stream2.wait_stream(current_stream)
    with torch.cuda.stream(stream2):
        flash_decode_stage2(
            q=q.view(calcu_shape1),
            k=cache_k,
            k_scale=cache_k_scale,
            v=cache_v,
            v_scale=cache_v_scale,
            Req_to_tokens=batch.req_to_token_indexs,
            B_req_idx=batch.b_req_idx,
            B_Seqlen=batch.b_seq_len,
            b_shared_seq_len=b_shared_seq_len,
            max_len_in_batch=batch.max_kv_seq_len,
            mid_out=mid_o,
            mid_out_logsumexp=mid_o_logexpsum,
            block_seq=BLOCK_SEQ,
        )

    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    flash_diverse_decode_stage3(
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=batch.b_seq_len,
        b_shared_seq_len=b_shared_seq_len,
        O=o_tensor.view(calcu_shape1),
        block_seq=BLOCK_SEQ,
    )
    return o_tensor
