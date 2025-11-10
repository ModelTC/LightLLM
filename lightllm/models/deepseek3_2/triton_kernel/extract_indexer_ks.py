import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_extract_indexer_ks(
    I_buffer,      # Input buffer [large_size, 1, 132] uint8
    SrcLoc,        # Source indices [req_size] int32/int64
    O_fp8,         # Output FP8 [req_size, 128] float8_e4m3fn
    O_scale,       # Output scale [req_size] float32
    stride_i_bs,
    stride_i_h,
    stride_i_d,
    stride_o_fp8_bs,
    stride_o_fp8_d,
    stride_o_scale_bs,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel to extract FP8 K values and their scales from an indexed buffer.
    
    This kernel is the inverse of destindex_copy_indexer_ks. It reads from a 
    compact buffer format where each entry contains:
    - Bytes 0-127: FP8 key values (128 bytes)
    - Bytes 128-131: Float32 scale (4 bytes)
    
    The source location for each output element is specified by SrcLoc.
    """
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load source index for this thread
    src_index = tl.load(SrcLoc + cur_index).to(tl.int64)
    
    # Load K_fp8 from I_buffer[:, 0, :128]
    i_k_ptrs = I_buffer + src_index * stride_i_bs + stride_i_d * offs_d
    k_fp8_as_uint8 = tl.load(i_k_ptrs)
    
    # Convert uint8 to fp8 through bitcast
    k_fp8 = k_fp8_as_uint8.to(tl.float8e4nv, bitcast=True)
    
    # Store K_fp8 to output
    o_k_ptrs = O_fp8 + cur_index * stride_o_fp8_bs + stride_o_fp8_d * offs_d
    tl.store(o_k_ptrs, k_fp8)
    
    # Load K_scale from I_buffer[:, 0, 128:132] (4 bytes for float32)
    # Load 4 bytes and reconstruct float32 (little-endian)
    i_scale_base_ptr = I_buffer + src_index * stride_i_bs + BLOCK_DMODEL * stride_i_d
    
    # Load 4 bytes individually and combine them into uint32
    byte0 = tl.load(i_scale_base_ptr + 0 * stride_i_d).to(tl.uint32)
    byte1 = tl.load(i_scale_base_ptr + 1 * stride_i_d).to(tl.uint32)
    byte2 = tl.load(i_scale_base_ptr + 2 * stride_i_d).to(tl.uint32)
    byte3 = tl.load(i_scale_base_ptr + 3 * stride_i_d).to(tl.uint32)
    
    # Combine bytes into uint32 (little-endian: byte0 is LSB)
    scale_as_uint32 = byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    
    # Bitcast uint32 to float32
    k_scale = scale_as_uint32.to(tl.float32, bitcast=True)
    
    # Store scale to output
    o_scale_ptr = O_scale + cur_index * stride_o_scale_bs
    tl.store(o_scale_ptr, k_scale)
    
    return


@torch.no_grad()
def extract_indexer_ks(I_buffer: torch.Tensor, SrcLoc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract FP8-quantized key values and their scales from indexed locations in a buffer.
    
    This function is the inverse operation of destindex_copy_indexer_ks. It's used in
    the DeepSeek-V3.2 NSA (Neighbor-aware Sparse Attention) mechanism to retrieve
    compressed key representations from a memory buffer.
    
    Args:
        I_buffer: [large_size, 1, 132] torch.uint8
            Input buffer containing packed FP8 keys and float32 scales.
            Format: [:, 0, :128] = FP8 keys, [:, 0, 128:132] = float32 scales
        SrcLoc: [req_size] torch.int32 or torch.int64
            Source indices to extract from the input buffer
    
    Returns:
        tuple containing:
            - K_fp8: [req_size, 128] torch.float8_e4m3fn
                FP8-quantized key values
            - K_scale: [req_size] torch.float32
                Quantization scales for each key
        
    Example:
        >>> i_buffer = torch.zeros(1024, 1, 132, dtype=torch.uint8).cuda()
        >>> src_loc = torch.tensor([10, 20, 30], dtype=torch.int32).cuda()
        >>> k_fp8, k_scale = extract_indexer_ks(i_buffer, src_loc)
        >>> # k_fp8.shape == [3, 128], k_scale.shape == [3]
    """
    req_size = SrcLoc.shape[0]
    head_dim = 128
    
    assert I_buffer.dtype == torch.uint8, f"Expected I_buffer dtype=uint8, got {I_buffer.dtype}"
    assert I_buffer.shape[2] == 132, f"Expected I_buffer last dim=132, got {I_buffer.shape[2]}"
    
    # Allocate output tensors
    O_fp8 = torch.empty((req_size, head_dim), dtype=torch.float8_e4m3fn, device=I_buffer.device)
    O_scale = torch.empty((req_size,), dtype=torch.float32, device=I_buffer.device)
    
    grid = (req_size,)
    num_warps = 1
    
    _fwd_kernel_extract_indexer_ks[grid](
        I_buffer,
        SrcLoc,
        O_fp8,
        O_scale,
        I_buffer.stride(0),
        I_buffer.stride(1),
        I_buffer.stride(2),
        O_fp8.stride(0),
        O_fp8.stride(1),
        O_scale.stride(0),
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    
    return O_fp8, O_scale


def test_extract_indexer_ks():
    """Test the extract_indexer_ks kernel against the copy kernel"""
    import torch.nn.functional as F
    from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
    
    print("=" * 80)
    print("Testing extract_indexer_ks")
    print("=" * 80)
    
    # Test parameters
    q_seq_len = 50
    head_dim = 128
    large_size = 1024
    dtype = torch.bfloat16
    fp8_type = torch.float8_e4m3fn
    
    # Create random indices for writing
    write_indices = torch.randint(0, large_size, (q_seq_len,), device="cuda", dtype=torch.int32).unique()
    actual_seq_len = len(write_indices)
    
    # Create input tensors
    k_bf16_original = torch.randn((actual_seq_len, head_dim), dtype=dtype, device="cuda")
    
    # Quantize to FP8
    k_abs_max = k_bf16_original.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_original = (k_abs_max / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_original = (k_bf16_original / k_abs_max).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    # Create buffer and write data using destindex_copy_indexer_ks
    buffer = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_original, k_scale_original, write_indices, buffer)
    
    # Now extract the data back using extract_indexer_ks
    k_fp8_extracted, k_scale_extracted = extract_indexer_ks(buffer, write_indices)
    
    # Verify FP8 values match
    fp8_match = torch.allclose(
        k_fp8_extracted.to(torch.float32), 
        k_fp8_original.to(torch.float32), 
        atol=0, rtol=0
    )
    
    # Verify scales match
    scale_match = torch.allclose(
        k_scale_extracted, 
        k_scale_original.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    # Check dequantized values
    k_dequant_extracted = k_fp8_extracted.to(dtype) * k_scale_extracted.unsqueeze(-1)
    cosine_sim = F.cosine_similarity(k_dequant_extracted, k_bf16_original, dim=-1).mean()
    
    print(f"Test with seq_len={actual_seq_len}, head_dim={head_dim}")
    print(f"  FP8 values match: {fp8_match}")
    print(f"  Scale values match: {scale_match}")
    print(f"  Cosine similarity after dequantization: {cosine_sim:.6f}")
    
    assert fp8_match, "FP8 values do not match!"
    assert scale_match, "Scale values do not match!"
    assert cosine_sim > 0.99, f"Cosine similarity too low: {cosine_sim}"
    
    print("✓ Basic test passed!")
    print()
    
    # Test with sequential indices
    print("Testing sequential indices...")
    write_indices_seq = torch.arange(20, device="cuda", dtype=torch.int32)
    k_bf16_seq = torch.randn((20, head_dim), dtype=dtype, device="cuda")
    k_abs_max_seq = k_bf16_seq.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_seq = (k_abs_max_seq / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_seq = (k_bf16_seq / k_abs_max_seq).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    buffer_seq = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_seq, k_scale_seq, write_indices_seq, buffer_seq)
    k_fp8_ext_seq, k_scale_ext_seq = extract_indexer_ks(buffer_seq, write_indices_seq)
    
    fp8_match_seq = torch.allclose(
        k_fp8_ext_seq.to(torch.float32), 
        k_fp8_seq.to(torch.float32), 
        atol=0, rtol=0
    )
    scale_match_seq = torch.allclose(
        k_scale_ext_seq, 
        k_scale_seq.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    print(f"  Sequential indices: FP8={fp8_match_seq}, Scale={scale_match_seq}")
    assert fp8_match_seq and scale_match_seq
    print("✓ Sequential test passed!")
    print()
    
    # Test with single element
    print("Testing single element...")
    write_idx_single = torch.tensor([42], device="cuda", dtype=torch.int32)
    k_bf16_single = torch.randn((1, head_dim), dtype=dtype, device="cuda")
    k_abs_max_single = k_bf16_single.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_single = (k_abs_max_single / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_single = (k_bf16_single / k_abs_max_single).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    buffer_single = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_single, k_scale_single, write_idx_single, buffer_single)
    k_fp8_ext_single, k_scale_ext_single = extract_indexer_ks(buffer_single, write_idx_single)
    
    fp8_match_single = torch.allclose(
        k_fp8_ext_single.to(torch.float32), 
        k_fp8_single.to(torch.float32), 
        atol=0, rtol=0
    )
    scale_match_single = torch.allclose(
        k_scale_ext_single, 
        k_scale_single.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    print(f"  Single element: FP8={fp8_match_single}, Scale={scale_match_single}")
    assert fp8_match_single and scale_match_single
    print("✓ Single element test passed!")
    print()
    
    # Test with larger batch to check performance characteristics
    print("Testing larger batch (performance check)...")
    write_indices_large = torch.randint(0, large_size * 10, (500,), device="cuda", dtype=torch.int32).unique()
    actual_large_len = len(write_indices_large)
    k_bf16_large = torch.randn((actual_large_len, head_dim), dtype=dtype, device="cuda")
    k_abs_max_large = k_bf16_large.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_large = (k_abs_max_large / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_large = (k_bf16_large / k_abs_max_large).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    buffer_large = torch.zeros((large_size * 10, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_large, k_scale_large, write_indices_large, buffer_large)
    
    # Warm up
    for _ in range(3):
        _ = extract_indexer_ks(buffer_large, write_indices_large)
    
    # Time it
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(100):
        k_fp8_ext_large, k_scale_ext_large = extract_indexer_ks(buffer_large, write_indices_large)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fp8_match_large = torch.allclose(
        k_fp8_ext_large.to(torch.float32), 
        k_fp8_large.to(torch.float32), 
        atol=0, rtol=0
    )
    scale_match_large = torch.allclose(
        k_scale_ext_large, 
        k_scale_large.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    print(f"  Large batch (size={actual_large_len}): FP8={fp8_match_large}, Scale={scale_match_large}")
    print(f"  Average time per call: {elapsed/100*1000:.3f} ms")
    assert fp8_match_large and scale_match_large
    print("✓ Large batch test passed!")
    print()
    
    print("=" * 80)
    print("All tests passed successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_extract_indexer_ks()
