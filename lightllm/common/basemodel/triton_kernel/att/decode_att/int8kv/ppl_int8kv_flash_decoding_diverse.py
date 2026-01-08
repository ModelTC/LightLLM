# 为 diverse mode 定制设计的 int8kv flash decoding attention 实现，可以实现更高效的多样性采样
import torch
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops
from lightllm.common.basemodel.infer_struct import InferStateInfo
from .ppl_int8kv_flash_decoding_diverse_stage1 import flash_decode_stage1
from .ppl_int8kv_flash_decoding_diverse_stage3 import flash_diverse_decode_stage3
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


def token_decode_attention_flash_decoding(
    q,
    infer_state: InferStateInfo,
    cache_k,
    cache_k_scale,
    cache_v,
    cache_v_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    shared_streams_dict={},
):
    if "stream1" not in shared_streams_dict:
        shared_streams_dict["stream1"] = torch.cuda.Stream()
    if "stream2" not in shared_streams_dict:
        shared_streams_dict["stream2"] = torch.cuda.Stream()

    stream1 = shared_streams_dict["stream1"]
    stream2 = shared_streams_dict["stream2"]

    q_head_num = q.shape[1]
    head_dim = q.shape[2]

    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_kv_seq_len = infer_state.max_kv_seq_len
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 2, head_dim], dtype=q.dtype, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 2], dtype=q.dtype, device="cuda"
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
            Req_to_tokens=infer_state.req_manager.req_to_token_indexs,
            B_req_idx=infer_state.b_req_idx,
            b_shared_seq_len=infer_state.b_shared_seq_len,
            b_mark_shared_group=infer_state.b_mark_shared_group,
            max_len_in_batch=infer_state.max_kv_seq_len,
            mid_out=mid_o,
            mid_out_logsumexp=mid_o_logexpsum,
            block_seq=BLOCK_SEQ,
            max_batch_group_size=get_diverse_max_batch_shared_group_size(),
        )
    stream2.wait_stream(current_stream)
    with torch.cuda.stream(stream2):
        light_ops.group8_int8kv_flashdecoding_diverse_stage2(
            BLOCK_SEQ,
            mid_o,
            mid_o_logexpsum,
            1.0 / (head_dim ** 0.5),
            q.view(calcu_shape1),
            cache_k,
            cache_k_scale,
            cache_v,
            cache_v_scale,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            infer_state.b_shared_seq_len,
            infer_state.max_kv_seq_len,
        )

    current_stream.wait_stream(stream1)
    current_stream.wait_stream(stream2)

    flash_diverse_decode_stage3(
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=infer_state.b_seq_len,
        b_shared_seq_len=infer_state.b_shared_seq_len,
        O=o_tensor.view(calcu_shape1),
        block_seq=BLOCK_SEQ,
    )
    return o_tensor
