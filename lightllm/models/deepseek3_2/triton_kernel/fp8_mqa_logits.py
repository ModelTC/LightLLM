import triton
import triton.language as tl
import torch


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr, KV_ptr, KVScale_ptr, Weights_ptr, MemIndex_ptr,
    CuSeqlenKs_ptr, CuSeqlenKe_ptr, Output_ptr,
    seq_len, seq_len_kv, num_heads, head_dim,
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_kv_pool, stride_kv_dim,
    stride_w_seq, stride_w_head,
    stride_o_seq, stride_o_kv,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the range of seq positions this block handles
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # Offset arrays for this block
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator for logits
    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Create masks
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len_kv
    
    # Load mem_indices for the KV positions
    mem_indices = tl.load(MemIndex_ptr + offs_n, mask=mask_n, other=0)
    
    # Load scales for K
    scales = tl.load(KVScale_ptr + mem_indices, mask=mask_n, other=1.0)
    
    # Loop over all heads
    for h in range(num_heads):
        # Load weights for this head
        weights = tl.load(Weights_ptr + offs_m * stride_w_seq + h * stride_w_head, 
                         mask=mask_m, other=0.0)
        
        # Initialize score accumulator for this head
        score = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # Loop over head_dim in blocks
        for d_block in range(tl.cdiv(head_dim, BLOCK_SIZE_D)):
            d_start = d_block * BLOCK_SIZE_D
            offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
            mask_d = offs_d < head_dim
            
            # Load Q for this head and dimension block
            # Q shape: (seq_len, num_heads, head_dim)
            q_ptrs = Q_ptr + offs_m[:, None] * stride_q_seq + h * stride_q_head + offs_d[None, :] * stride_q_dim
            mask_q = (offs_m[:, None] < seq_len) & mask_d[None, :]
            q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)
            
            # Load K for this dimension block
            # KV shape: (pool_size, head_dim) as FP8 data
            k_ptrs = KV_ptr + mem_indices[:, None] * stride_kv_pool + offs_d[None, :] * stride_kv_dim
            mask_k = mask_n[:, None] & mask_d[None, :]
            k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
            
            # Apply scale to K (scale is per-row of K)
            k = k * scales[:, None]
            
            # Compute partial dot product: q @ k.T
            # q: (BLOCK_SIZE_M, BLOCK_SIZE_D), k: (BLOCK_SIZE_N, BLOCK_SIZE_D)
            # score: (BLOCK_SIZE_M, BLOCK_SIZE_N)
            score += tl.dot(q, tl.trans(k))
        
        # Apply ReLU to score
        score = tl.maximum(score, 0.0)
        
        # Multiply by weights and accumulate to logits
        logits += score * weights[:, None]
    
    # Apply mask based on cu_seqlen_ks and cu_seqlen_ke
    mask_ks = tl.load(CuSeqlenKs_ptr + offs_m, mask=mask_m, other=0)
    mask_ke = tl.load(CuSeqlenKe_ptr + offs_m, mask=mask_m, other=seq_len_kv)
    
    mask_lo = offs_n[None, :] >= mask_ks[:, None]
    mask_hi = offs_n[None, :] < mask_ke[:, None]
    mask_valid = mask_lo & mask_hi & mask_m[:, None] & mask_n[None, :]
    
    # Apply mask (-inf for masked positions)
    logits = tl.where(mask_valid, logits, float('-inf'))
    
    # Store output
    out_ptrs = Output_ptr + offs_m[:, None] * stride_o_seq + offs_n[None, :] * stride_o_kv
    mask_out = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len_kv)
    tl.store(out_ptrs, logits, mask=mask_out)


def fp8_paged_mqa_logits(
    q: torch.Tensor, 
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor, 
    mem_index: torch.Tensor, 
    cu_seqlen_ks: torch.Tensor, 
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor = None
) -> torch.Tensor:
    seq_len, num_heads, head_dim = q.shape
    seq_len_kv = mem_index.shape[0]
    
    if out is None:
        output = torch.empty((seq_len, seq_len_kv), device=q.device, dtype=torch.float32)
    else:
        output = out
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_D = 128  
    
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M), triton.cdiv(seq_len_kv, BLOCK_SIZE_N))
    
    _fp8_paged_mqa_logits_kernel[grid](
        q, kv, kv_scale, weights, mem_index,
        cu_seqlen_ks, cu_seqlen_ke, output,
        seq_len, seq_len_kv, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        weights.stride(0), weights.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return output