import torch
import time
import pytest
import numpy as np
import torch.nn.functional as F
import flashinfer
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
)
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

if HAS_VLLM:
    scaled_fp8_quant = vllm_ops.scaled_fp8_quant
else:
    scaled_fp8_quant = None

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
def test_context_attention_fwd_flashinfer_fp8(batch, seqlen, q_heads, kv_heads, head_dim):
    Z, N_CTX, Q_HEADS, KV_HEADS, HEAD_DIM = batch, seqlen, q_heads, kv_heads, head_dim
    dtype = torch.bfloat16
    kv = torch.randn((Z * N_CTX, 2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    # for i in range(Z * N_CTX):
    #     kv[i] = torch.randn((2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda") * (i % 64 + 1)

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * (N_CTX // 2)
    rand_num = torch.randint_like(b_seq_len, high=(N_CTX // 2), dtype=torch.int32, device="cuda")
    b_seq_len += rand_num
    b_ready_cache_len = torch.zeros_like(b_seq_len, dtype=torch.int32, device="cuda")
    if N_CTX > 1:
        b_ready_cache_len = torch.randint_like(b_seq_len, high=(N_CTX - 1) // 2, dtype=torch.int32, device="cuda")
    b_req_idx = torch.randperm(Z, dtype=torch.int32).cuda()
    q_lens = b_seq_len - b_ready_cache_len
    q_start_loc = q_lens.cumsum(0) - q_lens
    kv_start_loc = b_seq_len.cumsum(0) - b_seq_len

    q = torch.randn((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.zeros((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o1 = torch.zeros((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    infer_state = LlamaInferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len
    infer_state.b_ready_cache_len = b_ready_cache_len
    infer_state.b_start_loc = q_start_loc

    context_attention_fwd(
        q,
        kv[:, :KV_HEADS, :],
        kv[:, KV_HEADS:, :],
        o,
        infer_state.b_req_idx,
        infer_state.b_start_loc,
        infer_state.b_seq_len,
        infer_state.b_ready_cache_len,
        infer_state.max_len_in_batch,
        req_to_token_indexs,
    )

    batch_size = Z
    head_dim = HEAD_DIM
    q_heads = Q_HEADS
    kv_heads = KV_HEADS
    page_size = 1
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8).to(0)
    q_starts = torch.zeros((Z + 1,)).int().cuda()
    q_starts[1:] = torch.cumsum(b_seq_len - b_ready_cache_len, dim=0)
    kv_starts = torch.zeros_like(q_starts)
    kv_starts[1:] = torch.cumsum(b_seq_len, dim=0)
    q_indptr = q_starts.int()
    kv_indptr = kv_starts.int()
    kv_indices = torch.arange(Z * N_CTX).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, kv_start_loc):
        kv_indices[start : start + sl] = req_to_token_indexs[b][:sl]
    kv_last_page_len_buffer = torch.empty(batch_size, device="cuda:0", dtype=torch.int32)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        qo_indptr_buf=q_indptr,
        paged_kv_indptr_buf=kv_indptr,
        paged_kv_indices_buf=kv_indices,
        paged_kv_last_page_len_buf=kv_last_page_len_buffer,
    )
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32)
    k_cache = kv[:, :KV_HEADS, :].contiguous()
    v_cache = kv[:, KV_HEADS:, :].contiguous()
    k, k_scale = scaled_fp8_quant(k_cache.view(1, -1))
    v, v_scale = scaled_fp8_quant(v_cache.view(1, -1))
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        q_data_type=q.dtype,
        kv_data_type=torch.float8_e4m3fn,
    )
    wrapper.run(
        q,
        (k.view(-1, 1, kv_heads, head_dim), v.view(-1, 1, kv_heads, head_dim)),
        k_scale=k_scale,
        v_scale=v_scale,
        out=o1,
        return_lse=False,
    )

    # assert torch.allclose(o, o1, atol=1e-2, rtol=2e-1)
    cos_sim1 = F.cosine_similarity(o, o1).mean()
    print(cos_sim1)
    assert cos_sim1 == 1


if __name__ == "__main__":
    test_context_attention_fwd_flashinfer_fp8(16, 1024, 28, 4, 128)
