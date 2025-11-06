import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_indexer_ks(
    k_fp8,
    k_scale,
    mem_index,
    buffer_fp8,
    buffer_scale,
    stride_k_fp8_bs,
    stride_k_fp8_h,
    stride_k_fp8_d,
    stride_k_scale_bs,
    stride_k_scale_h,
    stride_k_scale_d,
    stride_buffer_fp8_bs,
    stride_buffer_fp8_h,
    stride_buffer_fp8_d,
    stride_buffer_scale_bs,
    stride_buffer_scale_h,
    stride_buffer_scale_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(mem_index + cur_index).to(tl.int64)

    # Load k_fp8 data
    k_fp8_ptrs = k_fp8 + cur_index * stride_k_fp8_bs + stride_k_fp8_h * offs_h[:, None] + stride_k_fp8_d * offs_d[None, :]
    k_fp8_data = tl.load(k_fp8_ptrs, mask=offs_h[:, None] < head_num, other=0.0)

    # Load k_scale data
    k_scale_ptrs = k_scale + cur_index * stride_k_scale_bs + stride_k_scale_h * offs_h[:, None] + stride_k_scale_d * tl.arange(0, 1)[None, :]
    k_scale_data = tl.load(k_scale_ptrs, mask=offs_h[:, None] < head_num, other=0.0)

    # Store k_fp8 to buffer_fp8
    buffer_fp8_ptrs = buffer_fp8 + dest_index * stride_buffer_fp8_bs + stride_buffer_fp8_h * offs_h[:, None] + stride_buffer_fp8_d * offs_d[None, :]
    tl.store(buffer_fp8_ptrs, k_fp8_data, mask=offs_h[:, None] < head_num)

    # Store k_scale to buffer_scale
    buffer_scale_ptrs = buffer_scale + dest_index * stride_buffer_scale_bs + stride_buffer_scale_h * offs_h[:, None] + stride_buffer_scale_d * tl.arange(0, 1)[None, :]
    tl.store(buffer_scale_ptrs, k_scale_data, mask=offs_h[:, None] < head_num)


@torch.no_grad()
def destindex_copy_indexer_ks(k_fp8, k_scale, mem_index, buffer):
    seq_len = mem_index.shape[0]
    head_num = k_fp8.shape[1]
    k_fp8_dim = k_fp8.shape[2]  # Should be 128 for float8
    k_scale_dim = k_scale.shape[2]  # Should be 1

    assert k_fp8.shape[1] == k_scale.shape[1]
    assert k_fp8_dim == 128, f"k_fp8 dim should be 128, got {k_fp8_dim}"
    assert k_scale_dim == 1, f"k_scale dim should be 1, got {k_scale_dim}"
    assert buffer.shape[2] == 132, f"buffer dim should be 132, got {buffer.shape[2]}"  # 128 + 4 bytes

    # Reinterpret buffer as the appropriate types for storing
    buffer_fp8 = buffer[:, :, :128].view(torch.float8_e4m3fn)
    buffer_scale = buffer[:, :, 128:132].view(torch.float32)[:, :, :1]

    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_indexer_ks[grid](
        k_fp8,
        k_scale,
        mem_index,
        buffer_fp8,
        buffer_scale,
        k_fp8.stride(0),
        k_fp8.stride(1),
        k_fp8.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        buffer_fp8.stride(0),
        buffer_fp8.stride(1),
        buffer_fp8.stride(2),
        buffer_scale.stride(0),
        buffer_scale.stride(1),
        buffer_scale.stride(2),
        head_num,
        BLOCK_DMODEL=k_fp8_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test():
    import torch.nn.functional as F

    # Test parameters similar to the usage in nsa_indexer_layer_inder.py
    B, N_CTX, H, K_DIM = 4, 1024, 8, 128  # batch_size, seq_len, heads, k_dim
    seq_len = 50  # number of tokens to copy
    dtype_fp8 = torch.float8_e4m3fn
    dtype_scale = torch.float32

    # Create test data
    k_fp8 = torch.randn((seq_len, H, K_DIM), dtype=dtype_fp8).cuda()
    k_scale = torch.randn((seq_len, H, 1), dtype=dtype_scale).cuda()
    mem_index = torch.randint(0, B * N_CTX, (seq_len,), dtype=torch.int32).cuda()

    # Create buffer [total_tokens, heads, 132]
    buffer = torch.zeros((B * N_CTX, H, 132), dtype=torch.uint8).cuda()

    # Call the function
    destindex_copy_indexer_ks(k_fp8, k_scale, mem_index, buffer)

    # Verify results
    for i in range(seq_len):
        dest_idx = mem_index[i].item()
        # Check k_fp8 part
        stored_fp8 = buffer[dest_idx, :, :128].view(dtype_fp8)
        expected_fp8 = k_fp8[i]
        assert torch.allclose(stored_fp8, expected_fp8, atol=1e-6), f"FP8 mismatch at index {i}"

        # Check k_scale part
        stored_scale = buffer[dest_idx, :, 128:].view(dtype_scale)[:, :1]
        expected_scale = k_scale[i]
        assert torch.allclose(stored_scale, expected_scale, atol=1e-6), f"Scale mismatch at index {i}"

    print("All tests passed!")


if __name__ == "__main__":
    test()
