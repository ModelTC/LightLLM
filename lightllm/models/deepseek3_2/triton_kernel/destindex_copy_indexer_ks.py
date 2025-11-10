import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_ks(
    K_fp8,
    K_scale,
    DestLoc,
    O_buffer,
    stride_k_bs,
    stride_k_d,
    stride_scale_bs,
    stride_scale_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton kernel to copy FP8 K values and their scales to an indexed output buffer.
    
    This kernel reads FP8 key values (128 dims) and their float32 scale values,
    then writes them to a compact buffer format where each entry contains:
    - Bytes 0-127: FP8 key values (128 bytes)
    - Bytes 128-131: Float32 scale (4 bytes)
    
    The destination location for each source element is specified by DestLoc.
    """
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load destination index for this thread
    dest_index = tl.load(DestLoc + cur_index).to(tl.int64)
    
    # Load K_fp8 (128 values) and K_scale (1 value) from source
    k_fp8_ptrs = K_fp8 + cur_index * stride_k_bs + stride_k_d * offs_d
    k_fp8 = tl.load(k_fp8_ptrs)
    
    k_scale = tl.load(K_scale + cur_index * stride_scale_bs)
    
    # Store K_fp8 to O_buffer[:, 0, :128]
    # Convert fp8 to uint8 through bitcast for storage in uint8 buffer
    o_k_ptrs = O_buffer + dest_index * stride_o_bs + stride_o_d * offs_d
    k_fp8_as_uint8 = k_fp8.to(tl.uint8, bitcast=True)
    tl.store(o_k_ptrs, k_fp8_as_uint8)
    
    # Store K_scale to O_buffer[:, 0, 128:132] (4 bytes for float32)
    # Convert float32 scale to 4 uint8 bytes using bitcast and bit manipulation
    o_scale_ptr = O_buffer + dest_index * stride_o_bs + BLOCK_DMODEL * stride_o_d
    scale_as_uint32 = k_scale.to(tl.float32, bitcast=True).to(tl.uint32, bitcast=True)
    
    # Store each byte of the float32 scale (little-endian)
    for i in range(4):
        byte_val = ((scale_as_uint32 >> (i * 8)) & 0xFF).to(tl.uint8)
        tl.store(o_scale_ptr + i * stride_o_d, byte_val)
    
    return


@torch.no_grad()
def destindex_copy_indexer_ks(K_fp8: torch.Tensor, K_scale: torch.Tensor, DestLoc: torch.Tensor, O_buffer: torch.Tensor):
    """
    Copy FP8-quantized key values and their scales to indexed locations in a buffer.
    
    This function is used in the DeepSeek-V3.2 NSA (Neighbor-aware Sparse Attention)
    mechanism to store compressed key representations in a memory buffer. Each key
    is stored with its FP8 representation (128 bytes) followed by its float32 scale
    (4 bytes), for a total of 132 bytes per key.
    
    Args:
        K_fp8: [q_seq_len, 128] torch.fp8_e4m3fn
            FP8-quantized key values
        K_scale: [q_seq_len, 1] torch.float32
            Quantization scales for each key
        DestLoc: [q_seq_len] torch.int32
            Destination indices in the output buffer
        O_buffer: [large_size, 1, 132] torch.uint8
            Output buffer where keys and scales will be written.
            Must be a uint8 tensor to allow mixed-type storage.
            Format: [:, 0, :128] = FP8 keys, [:, 0, 128:132] = float32 scales

    Returns:
        None (modifies O_buffer in-place)
        
    Example:
        >>> k_fp8 = torch.randn(50, 128).to(torch.float8_e4m3fn).cuda()
        >>> k_scale = torch.randn(50, 1).cuda()
        >>> dest_loc = torch.randint(0, 1024, (50,), dtype=torch.int32).cuda()
        >>> o_buffer = torch.zeros(1024, 1, 132, dtype=torch.uint8).cuda()
        >>> destindex_copy_indexer_ks(k_fp8, k_scale, dest_loc, o_buffer)
        >>> # Now o_buffer[dest_loc] contains the packed k_fp8 and k_scale data
    """
    seq_len = DestLoc.shape[0]
    head_dim = K_fp8.shape[1]
    
    assert head_dim == 128, f"Expected head_dim=128, got {head_dim}"
    assert K_scale.shape[0] == seq_len
    assert O_buffer.shape[2] == 132, f"Expected O_buffer last dim=132, got {O_buffer.shape[2]}"
    
    grid = (seq_len,)
    num_warps = 1
    
    _fwd_kernel_destindex_copy_indexer_ks[grid](
        K_fp8,
        K_scale,
        DestLoc,
        O_buffer,
        K_fp8.stride(0),
        K_fp8.stride(1),
        K_scale.stride(0),
        K_scale.stride(1),
        O_buffer.stride(0),
        O_buffer.stride(1),
        O_buffer.stride(2),
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test_destindex_copy_indexer_ks():
    """Test the destindex_copy_indexer_ks kernel"""
    import torch.nn.functional as F
    
    print("=" * 80)
    print("Testing destindex_copy_indexer_ks")
    print("=" * 80)
    
    # Test parameters
    q_seq_len = 50
    head_dim = 128
    large_size = 1024
    dtype = torch.bfloat16
    fp8_type = torch.float8_e4m3fn
    
    # Create random destination indices
    dest_loc = torch.randint(0, large_size, (q_seq_len,), device="cuda", dtype=torch.int32).unique()
    actual_seq_len = len(dest_loc)
    
    # Create input tensors
    k_bf16 = torch.randn((actual_seq_len, head_dim), dtype=dtype, device="cuda")
    
    # Quantize to FP8
    k_abs_max = k_bf16.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale = (k_abs_max / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8 = (k_bf16 / k_abs_max).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    # Create output buffer (as uint8 to allow reinterpretation)
    o_buffer_uint8 = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    
    # Run kernel
    destindex_copy_indexer_ks(k_fp8, k_scale, dest_loc, o_buffer_uint8)
    
    # Extract results
    k_fp8_out = o_buffer_uint8[:, 0, :128].view(fp8_type)
    
    # Extract scale by reinterpreting 4 bytes as float32
    scale_bytes = o_buffer_uint8[:, 0, 128:132].contiguous()
    k_scale_out = scale_bytes.view(-1, 4).view(torch.float32).squeeze(-1)
    
    # Verify results at destination locations
    k_fp8_extracted = k_fp8_out[dest_loc]
    k_scale_extracted = k_scale_out[dest_loc]
    
    # Check FP8 values match
    fp8_match = torch.allclose(
        k_fp8_extracted.to(torch.float32), 
        k_fp8.to(torch.float32), 
        atol=0, rtol=0
    )
    
    # Check scales match
    scale_match = torch.allclose(
        k_scale_extracted, 
        k_scale.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    # Check dequantized values
    k_dequant_out = k_fp8_extracted.to(dtype) * k_scale_extracted.unsqueeze(-1)
    cosine_sim = F.cosine_similarity(k_dequant_out, k_bf16, dim=-1).mean()
    
    print(f"Test with seq_len={actual_seq_len}, head_dim={head_dim}")
    print(f"  FP8 values match: {fp8_match}")
    print(f"  Scale values match: {scale_match}")
    print(f"  Cosine similarity after dequantization: {cosine_sim:.6f}")
    
    assert fp8_match, "FP8 values do not match!"
    assert scale_match, "Scale values do not match!"
    assert cosine_sim > 0.99, f"Cosine similarity too low: {cosine_sim}"
    
    print("✓ Basic test passed!")
    print()
    
    # Test edge cases
    print("Testing edge cases...")
    
    # Test with sequential indices
    dest_loc_seq = torch.arange(20, device="cuda", dtype=torch.int32)
    k_bf16_seq = torch.randn((20, head_dim), dtype=dtype, device="cuda")
    k_abs_max_seq = k_bf16_seq.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_seq = (k_abs_max_seq / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_seq = (k_bf16_seq / k_abs_max_seq).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    o_buffer_seq = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_seq, k_scale_seq, dest_loc_seq, o_buffer_seq)
    
    k_fp8_out_seq = o_buffer_seq[:20, 0, :128].view(fp8_type)
    scale_bytes_seq = o_buffer_seq[:20, 0, 128:132].contiguous()
    k_scale_out_seq = scale_bytes_seq.view(-1, 4).view(torch.float32).squeeze(-1)
    
    fp8_match_seq = torch.allclose(
        k_fp8_out_seq.to(torch.float32), 
        k_fp8_seq.to(torch.float32), 
        atol=0, rtol=0
    )
    scale_match_seq = torch.allclose(
        k_scale_out_seq, 
        k_scale_seq.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    print(f"  Sequential indices test: FP8={fp8_match_seq}, Scale={scale_match_seq}")
    assert fp8_match_seq and scale_match_seq
    print("✓ Edge case tests passed!")
    print()
    
    # Test with single element
    print("Testing single element...")
    dest_loc_single = torch.tensor([42], device="cuda", dtype=torch.int32)
    k_bf16_single = torch.randn((1, head_dim), dtype=dtype, device="cuda")
    k_abs_max_single = k_bf16_single.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_single = (k_abs_max_single / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_single = (k_bf16_single / k_abs_max_single).clamp(
        torch.finfo(fp8_type).min, torch.finfo(fp8_type).max
    ).to(fp8_type)
    
    o_buffer_single = torch.zeros((large_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_single, k_scale_single, dest_loc_single, o_buffer_single)
    
    k_fp8_out_single = o_buffer_single[42:43, 0, :128].view(fp8_type)
    scale_bytes_single = o_buffer_single[42:43, 0, 128:132].contiguous()
    k_scale_out_single = scale_bytes_single.view(-1, 4).view(torch.float32).squeeze(-1)
    
    fp8_match_single = torch.allclose(
        k_fp8_out_single.to(torch.float32), 
        k_fp8_single.to(torch.float32), 
        atol=0, rtol=0
    )
    scale_match_single = torch.allclose(
        k_scale_out_single, 
        k_scale_single.squeeze(-1), 
        atol=1e-6, rtol=1e-5
    )
    
    print(f"  Single element test: FP8={fp8_match_single}, Scale={scale_match_single}")
    assert fp8_match_single and scale_match_single
    print("✓ Single element test passed!")
    print()
    
    print("=" * 80)
    print("All tests passed successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_destindex_copy_indexer_ks()