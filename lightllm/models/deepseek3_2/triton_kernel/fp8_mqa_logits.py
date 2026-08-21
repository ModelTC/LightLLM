import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr,
    stride_q_seq,
    stride_q_head,
    K_ptr,
    stride_k_pool,
    KScale_ptr,
    stride_kscale_pool,
    Weights_ptr,
    stride_w_seq,
    RaggedMemIndex_ptr,
    CuSeqlenKs_ptr,
    CuSeqlenKe_ptr,
    Out_ptr,
    stride_o_seq,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_q_index = tl.program_id(0)
    block_n_index = tl.program_id(1)

    ks = tl.load(CuSeqlenKs_ptr + cur_q_index)
    ke = tl.load(CuSeqlenKe_ptr + cur_q_index)
    kv_len = ke - ks
    start_n = block_n_index * BLOCK_N
    if start_n >= kv_len:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < kv_len
    offs_h = tl.arange(0, HEAD_NUM)
    offs_d = tl.arange(0, HEAD_DIM)

    mem_index = tl.load(RaggedMemIndex_ptr + ks + offs_n, mask=mask_n, other=0)

    # q: [HEAD_NUM, HEAD_DIM] fp8, k: [BLOCK_N, HEAD_DIM] fp8 gathered from the kv pool.
    q = tl.load(Q_ptr + cur_q_index * stride_q_seq + offs_h[:, None] * stride_q_head + offs_d[None, :])
    k = tl.load(K_ptr + mem_index[:, None] * stride_k_pool + offs_d[None, :], mask=mask_n[:, None], other=0.0)

    # relu commutes with the (positive) fp8 scales: q_scale is folded into weights by the
    # caller, k_scale is applied on the reduced column below.
    logits = tl.dot(q, tl.trans(k))  # [HEAD_NUM, BLOCK_N] fp32
    logits = tl.maximum(logits, 0.0)

    weights = tl.load(Weights_ptr + cur_q_index * stride_w_seq + offs_h)  # [HEAD_NUM]
    k_scale = tl.load(KScale_ptr + mem_index * stride_kscale_pool, mask=mask_n, other=0.0)  # [BLOCK_N]
    out = tl.sum(logits * weights[:, None], axis=0) * k_scale

    # row-compacted layout (matches deep_gemm.fp8_mqa_logits): row m's keys land at columns [0, ke-ks)
    tl.store(Out_ptr + cur_q_index * stride_o_seq + offs_n, out, mask=mask_n)


@torch.no_grad()
def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    indexer_k_buffer: torch.Tensor,
    weights: torch.Tensor,
    ragged_mem_index: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    max_kv_seq_len: int,
) -> torch.Tensor:
    """Indexer logits over the paged (token-granular) indexer-K pool.

    Row m holds k_scale_j * sum_h(weights[m, h] * relu(q[m, h] @ k_j)) for the ragged keys
    j in [cu_seqlen_ks[m], cu_seqlen_ke[m]), row-compacted at columns [0, ke-ks) — the same
    output contract as deep_gemm.fp8_mqa_logits(clean_logits=False) on the densely extracted
    K, but reading K directly from the kv pool through ragged_mem_index.

    q_fp8:            [q_token_num, head_num, 128] float8_e4m3fn
    indexer_k_buffer: [pool_size, 1, 132] uint8 (128B fp8 K + 4B fp32 scale per token)
    weights:          [q_token_num, head_num] float32 (q_scale folded in)
    ragged_mem_index / cu_seqlen_ks / cu_seqlen_ke: from gen_nsa_ks_ke
    """
    q_token_num, head_num, head_dim = q_fp8.shape
    assert head_dim == 128 and head_num in (16, 32, 64, 128)
    assert indexer_k_buffer.dtype == torch.uint8 and indexer_k_buffer.shape[2] == 132

    k_fp8 = indexer_k_buffer[:, 0, 0:128].view(dtype=torch.float8_e4m3fn)
    k_scale = indexer_k_buffer[:, 0, 128:132].view(dtype=torch.float32)

    logits = torch.empty((q_token_num, max_kv_seq_len), dtype=torch.float32, device=q_fp8.device)

    BLOCK_N = 128
    grid = (q_token_num, triton.cdiv(max_kv_seq_len, BLOCK_N))
    _fp8_paged_mqa_logits_kernel[grid](
        q_fp8,
        stride_q_seq=q_fp8.stride(0),
        stride_q_head=q_fp8.stride(1),
        K_ptr=k_fp8,
        stride_k_pool=k_fp8.stride(0),
        KScale_ptr=k_scale,
        stride_kscale_pool=k_scale.stride(0),
        Weights_ptr=weights,
        stride_w_seq=weights.stride(0),
        RaggedMemIndex_ptr=ragged_mem_index,
        CuSeqlenKs_ptr=cu_seqlen_ks,
        CuSeqlenKe_ptr=cu_seqlen_ke,
        Out_ptr=logits,
        stride_o_seq=logits.stride(0),
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return logits
