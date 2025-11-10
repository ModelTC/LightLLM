import torch
import triton
import triton.language as tl
import numpy


@triton.jit
def _fwd_kernel_extract_indexer_ks(
    buffer_fp8,
    buffer_scale,
    mem_index,
    k_fp8_out,
    k_scale_out,
    stride_buffer_fp8_bs,
    stride_buffer_fp8_h,
    stride_buffer_fp8_d,
    stride_buffer_scale_bs,
    stride_buffer_scale_h,
    stride_buffer_scale_d,
    stride_k_fp8_out_bs,
    stride_k_fp8_out_d,
    stride_k_scale_out_bs,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)

    # Load the memory index
    mem_idx = tl.load(mem_index + cur_index).to(tl.int64)

    # Load k_fp8 data from buffer_fp8[mem_idx, 0, :]
    offs_d = tl.arange(0, BLOCK_DMODEL)
    k_fp8_ptrs = buffer_fp8 + mem_idx * stride_buffer_fp8_bs + 0 * stride_buffer_fp8_h + offs_d * stride_buffer_fp8_d
    k_fp8_data = tl.load(k_fp8_ptrs)

    # Load k_scale data from buffer_scale[mem_idx, 0, 0]
    k_scale_ptr = buffer_scale + mem_idx * stride_buffer_scale_bs + 0 * stride_buffer_scale_h + 0 * stride_buffer_scale_d
    k_scale_data = tl.load(k_scale_ptr)

    # Store k_fp8 output
    k_fp8_out_ptrs = k_fp8_out + cur_index * stride_k_fp8_out_bs + offs_d * stride_k_fp8_out_d
    tl.store(k_fp8_out_ptrs, k_fp8_data)

    # Store k_scale output
    k_scale_out_ptr = k_scale_out + cur_index * stride_k_scale_out_bs
    tl.store(k_scale_out_ptr, k_scale_data)


@torch.no_grad()
def extract_indexer_ks(buffer, mem_index):
    """
    Extract k_fp8 and k_scale from the indexer memory buffer using Triton kernel.

    Args:
        buffer: Memory buffer of shape [total_tokens, heads, 132] with dtype uint8
        mem_index: Indices tensor of shape [seq_len] with dtype int32/int64

    Returns:
        k_fp8: Tensor of shape [seq_len, 128] with dtype float8_e4m3fn
        k_scale: Tensor of shape [seq_len] with dtype float32
    """
    seq_len = mem_index.shape[0]
    assert buffer.shape[2] == 132, f"buffer dim should be 132, got {buffer.shape[2]}"

    # Reinterpret buffer as the appropriate types for Triton
    buffer_fp8 = buffer[:, :, :128].view(torch.float8_e4m3fn)
    buffer_scale = buffer[:, :, 128:132].view(torch.float32)[:, :, :1]

    # Prepare output tensors
    k_fp8_out = torch.empty((seq_len, 128), dtype=torch.float8_e4m3fn, device=buffer.device)
    k_scale_out = torch.empty((seq_len,), dtype=torch.float32, device=buffer.device)

    BLOCK_DMODEL = 128
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_extract_indexer_ks[grid](
        buffer_fp8,
        buffer_scale,
        mem_index,
        k_fp8_out,
        k_scale_out,
        buffer_fp8.stride(0),
        buffer_fp8.stride(1),
        buffer_fp8.stride(2),
        buffer_scale.stride(0),
        buffer_scale.stride(1),
        buffer_scale.stride(2),
        k_fp8_out.stride(0),
        k_fp8_out.stride(1),
        k_scale_out.stride(0),
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )

    return k_fp8_out, k_scale_out


def test():
    # Test parameters similar to the usage in nsa_indexer_layer_inder.py
    B, N_CTX, H = 4, 1024, 1  # batch_size, seq_len, heads (always 1 for this)
    seq_len = 50  # number of tokens to extract
    dtype_fp8 = torch.float8_e4m3fn
    dtype_scale = torch.float32

    # Create test buffer [total_tokens, heads, 132] as uint8
    buffer = torch.zeros((B * N_CTX, H, 132), dtype=torch.uint8).cuda()

    # Fill with test data - simulate what destindex_copy_indexer_ks does
    test_indices = torch.randint(0, B * N_CTX, (seq_len,), dtype=torch.int32).cuda()
    # Generate fp8 data by converting from float32
    test_k_fp8_fp32 = torch.randn((seq_len, 128), dtype=torch.float32).cuda()
    test_k_fp8 = test_k_fp8_fp32.to(dtype_fp8)
    test_k_scale = torch.randn((seq_len,), dtype=dtype_scale).cuda()

    # Manually populate buffer as destindex_copy_indexer_ks would
    for i in range(seq_len):
        dest_idx = test_indices[i].item()
        # Store fp8 data
        buffer[dest_idx, 0, :128] = test_k_fp8[i].view(torch.uint8)
        # Store scale data (4 bytes) - need to convert float32 to bytes
        scale_bytes = test_k_scale[i].cpu().numpy().tobytes()
        scale_bytes_np = numpy.frombuffer(scale_bytes, dtype=numpy.uint8)
        buffer[dest_idx, 0, 128:132] = torch.from_numpy(scale_bytes_np).to(buffer.device)

    # Call our extraction function
    extracted_fp8, extracted_scale = extract_indexer_ks(buffer, test_indices)

    # Verify results
    print(f"Original k_fp8 shape: {test_k_fp8.shape}, dtype: {test_k_fp8.dtype}")
    print(f"Extracted k_fp8 shape: {extracted_fp8.shape}, dtype: {extracted_fp8.dtype}")
    print(f"Original k_scale shape: {test_k_scale.shape}, dtype: {test_k_scale.dtype}")
    print(f"Extracted k_scale shape: {extracted_scale.shape}, dtype: {extracted_scale.dtype}")

    # Check if extraction matches (convert fp8 to float32 for comparison)
    # Use higher tolerance for fp8 due to quantization precision
    fp8_match = torch.allclose(test_k_fp8_fp32, extracted_fp8.float(), atol=0.1, rtol=0.1)
    scale_match = torch.allclose(test_k_scale, extracted_scale, atol=1e-6)

    print(f"FP8 data matches: {fp8_match}")
    print(f"Scale data matches: {scale_match}")

    if fp8_match and scale_match:
        print("All tests passed!")
    else:
        print("Test failed!")
        if not fp8_match:
            print("First few fp8 values:")
            print(f"Original: {test_k_fp8_fp32[0, :5]}")
            print(f"Extracted: {extracted_fp8.float()[0, :5]}")
        if not scale_match:
            print(f"Max scale diff: {torch.max(torch.abs(test_k_scale - extracted_scale))}")


if __name__ == "__main__":
    test()
