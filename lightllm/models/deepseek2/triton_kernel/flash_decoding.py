import torch


def token_decode_attention_flash_decoding(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    infer_state,
    q_head_num,
    kv_lora_rank,
    q_rope_dim,
    qk_nope_head_dim,
    softmax_scale,
    out=None,
    alloc_tensor_func=torch.empty,
):
    if kv_lora_rank > 128:
        BLOCK_SEQ = 256 // (kv_lora_rank // 128)
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, kv_lora_rank)
    calcu_shape2 = (batch_size, q_head_num, q_rope_dim)

    from lightllm.models.deepseek2.triton_kernel.flash_decoding_stage1 import flash_decode_stage1
    from lightllm.models.deepseek2.triton_kernel.flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q_nope.shape, q_nope.dtype, q_nope.device) if out is None else out
    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, kv_lora_rank],
        dtype=torch.float32,
        device="cuda",
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
        qk_nope_head_dim,
        softmax_scale,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor


def torch_att(q, q_rope, kv, kv_rope, bs, seqlen, num_head, q_head_dim, rope_head_dim):
    import math

    xq = torch.cat([q, q_rope], dim=2).view(bs, 1, num_head, -1)
    xk = torch.cat([kv, kv_rope], dim=2).view(bs, seqlen, 1, -1)
    xv = kv.view(bs, seqlen, 1, -1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    # print(xq.shape, keys.transpose(2, 3).shape)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(q_head_dim + rope_head_dim)
    import torch.nn.functional as F

    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, q_head_dim)
    return output


def test():
    import torch
    import numpy as np
    from lightllm.common.basemodel import InferStateInfo
    from lightllm.common.req_manager import ReqManager

    Z, H, N_CTX, D_HEAD, ROPE_HEAD = 1, 6, 500, 128, 64
    dtype = torch.float16
    Z = 1
    q = torch.empty((Z, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z, H, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    req_to_token_indexs = torch.zeros((10, Z * N_CTX), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    Z = 1
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")

    b_seq_len[0] = N_CTX
    b_req_idx[0] = 0
    req_to_token_indexs[0][:N_CTX] = torch.tensor(np.arange(N_CTX), dtype=torch.int32).cuda()

    torch_out = torch_att(q, q_rope, kv, kv_rope, Z, N_CTX, H, D_HEAD, ROPE_HEAD)

    infer_state = InferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = 500
    infer_state.req_manager = ReqManager(10, 500, None)
    infer_state.req_manager.req_to_token_indexs = req_to_token_indexs
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len
    infer_state.max_len_in_batch = max_input_len

    o = token_decode_attention_flash_decoding(q, q_rope, kv, kv_rope, infer_state, H, D_HEAD, ROPE_HEAD)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test()
