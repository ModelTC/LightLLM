import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_copy_indexer_ks(
    buffer,  # [large_size, 1, 132] uint8
    src_loc,  # [copy_len] int32/int64 - source indices
    dest_loc,  # [copy_len] int32/int64 - destination indices
    stride_bs,
    stride_h,
    stride_d,
    BLOCK_KV: tl.constexpr,  # = 128 (FP8 data)
    BLOCK_SCALE: tl.constexpr,  # = 4 (scale data)
):
    """
    Triton kernel to copy indexer_ks data from source locations to destination locations.

    This kernel copies 132-byte indexer_ks entries (128 bytes FP8 key + 4 bytes float32 scale)
    from source positions to destination positions within the same buffer.

    Args:
        buffer: Shared buffer containing indexer_ks data [large_size, 1, 132] uint8
        src_loc: Source indices to copy from [copy_len]
        dest_loc: Destination indices to copy to [copy_len]
        stride_bs, stride_h, stride_d: Strides for the buffer
        BLOCK_KV: Size of FP8 key data (128 bytes)
        BLOCK_SCALE: Size of scale data (4 bytes)
    """
    cur_index = tl.program_id(0)
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_scale = tl.arange(0, BLOCK_SCALE)

    # Load source and destination indices
    src_index = tl.load(src_loc + cur_index).to(tl.int64)
    dest_index = tl.load(dest_loc + cur_index).to(tl.int64)

    # Copy FP8 key data (128 bytes)
    src_kv_ptrs = buffer + src_index * stride_bs + stride_d * offs_kv
    dest_kv_ptrs = buffer + dest_index * stride_bs + stride_d * offs_kv
    kv_data = tl.load(src_kv_ptrs)
    tl.store(dest_kv_ptrs, kv_data)

    # Copy scale data (4 bytes at offset 128)
    src_scale_base = buffer + src_index * stride_bs + BLOCK_KV * stride_d
    dest_scale_base = buffer + dest_index * stride_bs + BLOCK_KV * stride_d
    scale_data = tl.load(src_scale_base + offs_scale * stride_d)
    tl.store(dest_scale_base + offs_scale * stride_d, scale_data)

    return


@torch.no_grad()
def copy_indexer_ks(
    buffer: torch.Tensor,
    src_loc: torch.Tensor,
    dest_loc: torch.Tensor,
):
    """
    Copy indexer_ks data from source positions to destination positions.

    This function is used to copy cached tokens' indexer_ks data to new locations
    after prefix cache matching. It ensures that the indexer_ks buffer stays
    consistent with the KV cache buffer.

    Args:
        buffer: [large_size, 1, 132] torch.uint8
            Buffer containing indexer_ks data (same buffer for src and dest)
        src_loc: [copy_len] torch.int32 or torch.int64
            Source indices in buffer (old positions)
        dest_loc: [copy_len] torch.int32 or torch.int64
            Destination indices in buffer (new positions)

    Returns:
        None (modifies buffer in-place)

    Example:
        >>> buffer = torch.zeros((1024, 1, 132), dtype=torch.uint8).cuda()
        >>> old_pos = torch.tensor([100, 101, 102], dtype=torch.int32).cuda()
        >>> new_pos = torch.tensor([200, 201, 202], dtype=torch.int32).cuda()
        >>> copy_indexer_ks(buffer, old_pos, new_pos)
        # Data from positions [100, 101, 102] is now copied to [200, 201, 202]
    """
    copy_len = src_loc.shape[0]
    block_kv = 128  # FP8 key data size
    block_scale = 4  # Float32 scale size

    assert (
        src_loc.shape[0] == dest_loc.shape[0]
    ), f"src_loc and dest_loc must have same length: {src_loc.shape[0]} != {dest_loc.shape[0]}"
    assert (
        buffer.shape[2] == block_kv + block_scale
    ), f"Expected buffer last dim={block_kv + block_scale}, got {buffer.shape[2]}"
    assert buffer.dtype == torch.uint8, f"Expected buffer dtype=uint8, got {buffer.dtype}"

    grid = (copy_len,)
    num_warps = 1

    _fwd_kernel_copy_indexer_ks[grid](
        buffer,
        src_loc,
        dest_loc,
        buffer.stride(0),
        buffer.stride(1),
        buffer.stride(2),
        BLOCK_KV=block_kv,
        BLOCK_SCALE=block_scale,
        num_warps=num_warps,
        num_stages=1,
    )

    return


def test_copy_indexer_ks():
    """Test the copy_indexer_ks kernel"""
    import torch.nn.functional as F
    from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks
    from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import extract_indexer_ks

    print("=" * 80)
    print("Testing copy_indexer_ks")
    print("=" * 80)

    # Test parameters
    cached_len = 20
    buffer_size = 1024
    head_dim = 128
    dtype = torch.bfloat16
    fp8_type = torch.float8_e4m3fn

    # Create indexer_ks data
    k_bf16 = torch.randn((cached_len, head_dim), dtype=dtype, device="cuda")

    # Quantize to FP8
    k_abs_max = k_bf16.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale = (k_abs_max / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8 = (k_bf16 / k_abs_max).clamp(torch.finfo(fp8_type).min, torch.finfo(fp8_type).max).to(fp8_type)

    # Write to old positions
    old_positions = torch.arange(100, 100 + cached_len, dtype=torch.int32, device="cuda")
    buffer = torch.zeros((buffer_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8, k_scale, old_positions, buffer)

    # Copy to new positions
    new_positions = torch.arange(200, 200 + cached_len, dtype=torch.int32, device="cuda")
    copy_indexer_ks(buffer, old_positions, new_positions)

    # Verify data at new positions matches original
    k_fp8_extracted, k_scale_extracted = extract_indexer_ks(buffer, new_positions)

    fp8_match = torch.allclose(k_fp8_extracted.to(torch.float32), k_fp8.to(torch.float32), atol=0, rtol=0)

    scale_match = torch.allclose(k_scale_extracted, k_scale.squeeze(-1), atol=1e-6, rtol=1e-5)

    # Check dequantized values
    k_dequant_extracted = k_fp8_extracted.to(dtype) * k_scale_extracted.unsqueeze(-1)
    cosine_sim = F.cosine_similarity(k_dequant_extracted, k_bf16, dim=-1).mean()

    print(f"Cached tokens: {cached_len}, Head dim: {head_dim}")
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
    old_pos_seq = torch.arange(20, dtype=torch.int32, device="cuda")
    new_pos_seq = torch.arange(200, 220, dtype=torch.int32, device="cuda")

    k_bf16_seq = torch.randn((20, head_dim), dtype=dtype, device="cuda")
    k_abs_max_seq = k_bf16_seq.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_seq = (k_abs_max_seq / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_seq = (k_bf16_seq / k_abs_max_seq).clamp(torch.finfo(fp8_type).min, torch.finfo(fp8_type).max).to(fp8_type)

    buffer_seq = torch.zeros((buffer_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_seq, k_scale_seq, old_pos_seq, buffer_seq)
    copy_indexer_ks(buffer_seq, old_pos_seq, new_pos_seq)

    k_fp8_ext_seq, k_scale_ext_seq = extract_indexer_ks(buffer_seq, new_pos_seq)

    fp8_match_seq = torch.allclose(k_fp8_ext_seq.to(torch.float32), k_fp8_seq.to(torch.float32), atol=0, rtol=0)
    scale_match_seq = torch.allclose(k_scale_ext_seq, k_scale_seq.squeeze(-1), atol=1e-6, rtol=1e-5)

    print(f"  Sequential indices: FP8={fp8_match_seq}, Scale={scale_match_seq}")
    assert fp8_match_seq and scale_match_seq
    print("✓ Sequential test passed!")
    print()

    # Test with single element
    print("Testing single element...")
    old_pos_single = torch.tensor([42], dtype=torch.int32, device="cuda")
    new_pos_single = torch.tensor([424], dtype=torch.int32, device="cuda")

    k_bf16_single = torch.randn((1, head_dim), dtype=dtype, device="cuda")
    k_abs_max_single = k_bf16_single.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    k_scale_single = (k_abs_max_single / torch.finfo(fp8_type).max).to(torch.float32)
    k_fp8_single = (
        (k_bf16_single / k_abs_max_single).clamp(torch.finfo(fp8_type).min, torch.finfo(fp8_type).max).to(fp8_type)
    )

    buffer_single = torch.zeros((buffer_size, 1, 132), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_ks(k_fp8_single, k_scale_single, old_pos_single, buffer_single)
    copy_indexer_ks(buffer_single, old_pos_single, new_pos_single)

    k_fp8_ext_single, k_scale_ext_single = extract_indexer_ks(buffer_single, new_pos_single)

    fp8_match_single = torch.allclose(
        k_fp8_ext_single.to(torch.float32), k_fp8_single.to(torch.float32), atol=0, rtol=0
    )
    scale_match_single = torch.allclose(k_scale_ext_single, k_scale_single.squeeze(-1), atol=1e-6, rtol=1e-5)

    print(f"  Single element: FP8={fp8_match_single}, Scale={scale_match_single}")
    assert fp8_match_single and scale_match_single
    print("✓ Single element test passed!")
    print()

    print("=" * 80)
    print("All tests passed successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_copy_indexer_ks()
