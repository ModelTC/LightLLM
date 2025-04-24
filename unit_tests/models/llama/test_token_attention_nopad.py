import torch
import time
import pytest
import numpy as np
import torch.nn.functional as F
import flashinfer
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.models.llama.triton_kernel.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ref_token_attention_nopad(q, k, v, o, q_h, h_dim, infer_state):
    from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd

    total_token_num = infer_state.total_token_num
    batch_size = infer_state.batch_size
    calcu_shape1 = (batch_size, q_h, h_dim)

    att_m_tensor = torch.empty((q_h, total_token_num), dtype=torch.float32).cuda()

    token_att_fwd(
        q.view(calcu_shape1),
        k,
        att_m_tensor,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_start_loc,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
    )

    from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import (
        token_softmax_reducev_fwd,
    )

    token_softmax_reducev_fwd(
        att_m_tensor,
        v,
        o,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_start_loc,
        infer_state.b_seq_len,
    )
    return o


@pytest.mark.parametrize(
    "batch, seqlen, q_heads, kv_heads, head_dim",
    [
        (a, b, c, d, e)
        for a in [1, 16, 32, 128, 512]
        for b in [16, 32, 512, 1024]
        for c in [28]
        for d in [4]
        for e in [128]
    ],
)
def test_context_attention_fwd(batch, seqlen, q_heads, kv_heads, head_dim):
    Z, N_CTX, Q_HEADS, KV_HEADS, HEAD_DIM = batch, seqlen, q_heads, kv_heads, head_dim
    dtype = torch.bfloat16
    q = torch.randn((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    kv = torch.randn((Z * N_CTX, 2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * N_CTX
    b_start_loc = torch.arange(Z).cuda().int() * N_CTX
    b_req_idx = torch.randperm(Z, dtype=torch.int32).cuda()

    o = torch.zeros((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o1 = torch.zeros((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    infer_state = LlamaInferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
    infer_state.req_manager = ReqManager(Z, N_CTX, None)
    infer_state.req_manager.req_to_token_indexs = req_to_token_indexs
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len
    infer_state.b_start_loc = b_start_loc

    ref_token_attention_nopad(
        q,
        kv[:, :KV_HEADS, :],
        kv[:, KV_HEADS:, :],
        o,
        Q_HEADS,
        HEAD_DIM,
        infer_state,
    )
    # gqa_decode_attention_fwd(
    #     q,
    #     kv[:,:KV_HEADS,:],
    #     kv[:,KV_HEADS:,:],
    #     o,
    #     infer_state.req_manager.req_to_token_indexs,
    #     infer_state.b_req_idx,
    #     infer_state.b_seq_len,
    # )

    batch_size = Z
    head_dim = HEAD_DIM
    q_heads = Q_HEADS
    kv_heads = KV_HEADS
    page_size = 1
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8).to(0)
    kv_starts = torch.zeros((Z + 1,)).int().cuda()
    kv_starts[1:] = torch.cumsum(b_seq_len, dim=0)
    kv_indptr = kv_starts
    kv_indices = torch.arange(Z * N_CTX).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        kv_indices[start : start + sl] = req_to_token_indexs[b][:sl]
    kv_last_page_len_buffer = torch.empty(batch_size, device="cuda:0", dtype=torch.int32)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=kv_indptr,
        paged_kv_indices_buffer=kv_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len_buffer,
    )
    kv_last_page_len_buffer = torch.full((batch_size,), page_size, dtype=torch.int32)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len_buffer,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        non_blocking=True,
    )
    kv = kv.unsqueeze(1)
    wrapper.run(q, (kv[:, :, :KV_HEADS, :], kv[:, :, KV_HEADS:, :]), out=o1, return_lse=False)

    cos_sim1 = F.cosine_similarity(o, o1).mean()
    assert cos_sim1 == 1.0
