import torch
import time
import pytest
import numpy as np
import torch.nn.functional as F
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kv_quantize_per_head_fp8(kv_buffer: torch.Tensor, seq_lens):
    device = kv_buffer.device
    B = seq_lens.size(0)
    min_fp8 = torch.finfo(torch.float8_e4m3fn).min
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    _, S_max, H, D = kv_buffer.shape
    seq_range = torch.arange(S_max, device=device)[None, :]
    valid_mask = (seq_range < seq_lens[:, None]).view(B, S_max, 1, 1)
    masked = kv_buffer * valid_mask
    max_per_bh = masked.float().abs().amax(dim=(1, 3))  # [B, H]
    scales = torch.where(max_per_bh > 0, max_per_bh / max_fp8, torch.ones_like(max_per_bh))
    scales_exp = scales.view(B, 1, H, 1)
    q = (kv_buffer / scales_exp).clamp(min_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q, scales


def ref_token_attention_nopad(q, k, v, o, q_h, h_dim, infer_state, req_to_token_indexs):
    from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd

    total_token_num = infer_state.total_token_num
    batch_size = infer_state.batch_size
    calcu_shape1 = (batch_size, q_h, h_dim)

    att_m_tensor = torch.empty((q_h, total_token_num), dtype=torch.float32).cuda()

    token_att_fwd(
        q.view(calcu_shape1),
        k,
        att_m_tensor,
        req_to_token_indexs,
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
        req_to_token_indexs,
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
def test_token_attention_nopad_fa3_fp8(batch, seqlen, q_heads, kv_heads, head_dim):
    Z, N_CTX, Q_HEADS, KV_HEADS, HEAD_DIM = batch, seqlen, q_heads, kv_heads, head_dim
    dtype = torch.bfloat16
    q = torch.randn((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    kv = torch.randn((Z * N_CTX, 2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    # for i in range(Z * N_CTX):
    #     kv[i] = torch.randn((2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda") * (i % 10 + 1)

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * (N_CTX // 2)
    rand_num = torch.randint_like(b_seq_len, high=(N_CTX // 2), dtype=torch.int32, device="cuda")
    b_seq_len += rand_num
    b_start_loc = b_seq_len.cumsum(0) - b_seq_len
    b_req_idx = torch.randperm(Z, dtype=torch.int32).cuda()

    o = torch.zeros((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o1 = torch.zeros((Z, Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    infer_state = LlamaInferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
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
        req_to_token_indexs,
    )
    # gqa_decode_attention_fwd(
    #     q,
    #     kv[:,:KV_HEADS,:],
    #     kv[:,KV_HEADS:,:],
    #     o,
    #     req_to_token_indexs,
    #     infer_state.b_req_idx,
    #     infer_state.b_seq_len,
    # )

    batch_size = Z
    head_dim = HEAD_DIM
    q_heads = Q_HEADS
    kv_heads = KV_HEADS
    kv_starts = torch.zeros((Z + 1,)).int().cuda()
    kv_starts[1:] = torch.cumsum(b_seq_len, dim=0)
    q_starts = torch.arange(0, Z + 1).int().cuda()
    page_table = torch.empty((batch_size, N_CTX), dtype=torch.int32).to(0)
    page_table.copy_(req_to_token_indexs[b_req_idx, :N_CTX])

    k_cache = kv[:, :KV_HEADS, :].contiguous()
    v_cache = kv[:, KV_HEADS:, :].contiguous()
    # o1 = flash_attn_with_kvcache(
    #     q=q,
    #     k_cache=k_cache[page_table].view(-1, N_CTX, kv_heads, head_dim),
    #     v_cache=v_cache[page_table].view(-1, N_CTX, kv_heads, head_dim),
    #     # page_table=page_table,
    #     cache_seqlens=infer_state.b_seq_len,
    #     cu_seqlens_q=q_starts,
    #     cu_seqlens_k_new=kv_starts,
    #     max_seqlen_q=1,
    #     causal=False,
    #     window_size=(-1, -1),
    #     softcap=0.0,
    #     return_softmax_lse=False,
    # )

    q, q_scale = scaled_fp8_quant(q.view(batch_size * kv_heads, -1), use_per_token_if_dynamic=True)
    k, k_scale = kv_quantize_per_head_fp8(k_cache[page_table], b_seq_len)
    v, v_scale = kv_quantize_per_head_fp8(v_cache[page_table], b_seq_len)
    o1 = flash_attn_with_kvcache(
        q=q.view(-1, q_heads, head_dim),
        k_cache=k.view(-1, N_CTX, kv_heads, head_dim),
        v_cache=v.view(-1, N_CTX, kv_heads, head_dim),
        # page_table=page_table,
        cache_seqlens=infer_state.b_seq_len,
        cu_seqlens_q=q_starts,
        cu_seqlens_k_new=kv_starts,
        max_seqlen_q=1,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        q_descale=q_scale.view(batch_size, kv_heads),
        k_descale=k_scale.view(batch_size, kv_heads),
        v_descale=v_scale.view(batch_size, kv_heads),
        return_softmax_lse=False,
    )

    # assert torch.allclose(o, o1, atol=1e-1, rtol=1e-1)
    cos_sim1 = F.cosine_similarity(o, o1).mean()
    print(cos_sim1)
    assert cos_sim1 == 1


if __name__ == "__main__":
    test_token_attention_nopad_fa3_fp8(16, 16384, 28, 4, 128)
