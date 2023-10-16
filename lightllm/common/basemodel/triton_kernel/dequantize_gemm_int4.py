import time

import torch

import triton
import triton.language as tl


@triton.autotune(
	configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), 
    ],
	key=['M', 'N', 'K', 'NO_GROUPS'],
)
@triton.jit
def matmul4_kernel(
	a_ptr, b_ptr, c_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_am, stride_ak,
	stride_bk, stride_bn,
	stride_cm, stride_cn,
	stride_scales_g, stride_scales_n,
	stride_zeros_g, stride_zeros_n,
	groupsize, NO_GROUPS: tl.constexpr,
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N//8) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is K // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    bits = 4
    infearure_per_bits = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    # zeros_ptrs is set up such that it repeats elements along the N axis 8 times
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    # shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    # If G == 1, scales and zeros are the same for all K, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
        # Unpack zeros
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        # zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16
        zeros = zeros * scales
    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
    	a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    	b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated	
    	if not NO_GROUPS:
    		g_id = k // (groupsize // BLOCK_SIZE_K)
    		ptr = scales_ptrs + g_id * stride_scales_g
    		scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
    		ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
    		zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
    		# Unpack zeros
    		zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
    		zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16	
    	# Now we need to unpack b (which is 4-bit values) into 32-bit values
    	b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
    	b = b * scales[None, :] - zeros[None, :]  # Scale and shift
    	# print("data type", a, b)
    	accumulator += tl.dot(a, b)
    	a_ptrs += BLOCK_SIZE_K * stride_ak
    	b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(tl.float16)  
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_dequantize_int4_gptq(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size, output=None) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = A x B + bias.
	Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

	A is of shape (..., K) float16
	qweight is of shape (K//8, N) int32
	scales is of shape (G, N) float16
	qzeros is of shape (G, N//8) int32
	bias is of shape (1, N) float16

	groupsize is the number of infeatures in each group.
	G = K // groupsize

	Returns C of shape (..., N) float16
	"""
	assert x.shape[-1] == (qweight.shape[0] * 8), "A must be a multiple of 8 in the last dimension"
	assert x.is_contiguous(), "A must be contiguous"

	M, K = x.shape
	N = qweight.shape[1]
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# # This is based on the possible BLOCK_SIZE_Ns
	# assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# # This is based on the possible BLOCK_SIZE_Ks
	# assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

	# output = torch.empty((M, N), device='cuda', dtype=torch.float16)
	if output is None:
		inplace = False
		output = torch.empty((M, N), device=x.device, dtype=torch.float16)
	else:
		inplace = True

	grid = lambda META: (
		triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	matmul4_kernel[grid](
		x, qweight, output,
		scales, qzeros,
		M, N, K,
		x.stride(0), x.stride(1),
		qweight.stride(0), qweight.stride(1),
		output.stride(0), output.stride(1),
		scales.stride(0), scales.stride(1),
		qzeros.stride(0), qzeros.stride(1),
		group_size, group_size == K,
    )
	# return output
	if not inplace:
		return output


@triton.autotune(
	configs=[
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
	    
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
		
 ],
	key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # [BLOCK_K, BLOCK_N] but repeated 8 times in N
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    # tl.static_print("shape", a_ptrs, b_ptrs, bs_ptrs, bzp_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B.
        # [BLOCK_K, BLOCK_N] but repeated group_size times in K 
        bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
            + offs_bn[None, :] * stride_bsn
        # [BLOCK_K, BLOCK_N] but repeated in K and N
        bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
            + (offs_bn[None, :] // 8) * stride_bzpn
        b_shift_bits = (offs_k[:, None] % 8) * 4 # assert BLOCK_SIZE_K % 8 == 0
        bzp_shift_bits = (offs_bn[None, :] % 8) * 4
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        bs = tl.load(bs_ptrs)
        bzp = tl.load(bzp_ptrs)
        # We accumulate along the K dimension.
        int_b = (b >> b_shift_bits) & 0xF
        int_bzp = (bzp >> bzp_shift_bits) & 0xF
        b = ((int_b - int_bzp) * bs).to(tl.float16)
        accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)  # assert BLOCK_SIZE_K % 8 == 0
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    """
    """
    assert x.is_contiguous(), "A must be contiguous"
    assert qweight.is_contiguous(), "B must be contiguous"  
    M, K = x.shape
    N = scales.shape[1]
    if output is None:
    	output = torch.zeros((M, N), device=x.device, dtype=torch.float16)  
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
    	x, qweight, output,
    	scales, qzeros,
    	M, N, K,
    	x.stride(0), x.stride(1),
    	qweight.stride(0), qweight.stride(1),
    	output.stride(0), output.stride(1),
    	scales.stride(0), scales.stride(1),
    	qzeros.stride(0), qzeros.stride(1),
    	group_size,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
    We should assert BLOCK_SIZE_N % 8 == 0.
    weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    fpb_offs = offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
    b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
    bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
    b_shift = (offs_k[:, None] % 8) * 4
    bzp_shift = (offs_n[None, :] % 8) * 4
    fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
    tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)


def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    Kw, N = b.shape
    K = Kw * 8
    fp_b = torch.ones((K, N), device=device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b


def matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size=128, out=None):
    """
    Matmul dequantize int4 s1 dequantize weight to `fp_b` and do fp16 torch.mm,
    this is for `prefill` stage, since weight size is fixed so is dequantize overhead,
    perfill stage have more tokens to amortize dequant cost.
    """
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        # Allocates output.
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=out)
    fp_b = None
    return out


def quantize_int4(weight, group_size=128):
    # Weight shape: [H1 // 8, H2]
    # Scale shape: [H1 // group_size, H2]
    # zero_pint shape: [H1 // group_size, H2 // 8]
    weight = weight.transpose(1, 0)
    h1, h2 = weight.shape
    assert h1 % 8 == 0 and h2 % 8 == 0, "H1 {} H2 {}".format(h1, h2)
    assert h2 % group_size == 0, "H1 {} H2 {}".format(h1, h2)

    weight = weight.contiguous().view(-1, group_size).cuda()
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    scale = weight_range / (2 ** 4 - 1)
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    weight = (weight / scale + zero_point).round().clamp(0, 15).to(torch.int32).view(h1, h2)
    int_weight = torch.empty(h1, h2 // 8).to(torch.int32).to(weight.device)
    int_zero_point = torch.zeros(h1 // 8, h2 // group_size).to(torch.int32).to(weight.device)
    zero_point = zero_point.view(h1, -1)
    scale = scale.view(h1, -1)
    # pack 8 int4 in an int32 number.
    # Weight pack in row.
    for pack in range(0, h2, 8):
        for i in range(8):
            int_weight[:, pack // 8] += weight[:, pack + i] << (i * 4)
    # zero point pack in col.
    for pack in range(0, h1, 8):
        for i in range(8):
            int_zero_point[pack // 8, :] += zero_point[pack + i, :] << (i * 4)
    '''
    fp_weight = torch.zeros(h1, h2).half().to(weight.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_weight[pack * 8 + i, :] = \
                ((int_weight[pack, :] << (28 - i * 4) >> 28) + 16) % 16
    print((fp_weight - weight).abs().sum())

    fp_zp = torch.zeros(zero_point.shape).half().to(zero_point.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zp[pack * 8 + i, :] = \
                (int_zero_point[pack, :] >> (i * 4)) & 15

    print((fp_zp - zero_point).abs().sum())
    '''
    weight = None
    return int_weight.transpose(1, 0).contiguous(), scale.transpose(1, 0).contiguous(), int_zero_point.transpose(1, 0).contiguous()


def unpack_int4(weight, scale, zp):
    """
    Test function to verify quantize int4 is correct.
    Will not be used in model inference.
    """
    weight = weight.transpose(1, 0)
    scale = scale.transpose(1, 0)
    zp = zp.transpose(1, 0)
    h1, h2 = weight.shape
    group_size = h2 * 8 // scale.shape[1]
    group_num = scale.shape[1]
    fp_weight = torch.zeros(h1, h2 * 8).half().to(weight.device)
    fp_zero_point = torch.zeros(h1, group_num).to(weight.device)
    for pack in range(0, h2):
        for i in range(8):
            fp_weight[:, pack * 8 + i] = (weight[:, pack] >> (i * 4)) & 0xF
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zero_point[pack * 8 + i, :] = (zp[pack, :] >> (i * 4)) & 0xF
    for g in range(group_num):
        fp_weight[:, g * group_size:(g + 1) * group_size] = (fp_weight[:, g * group_size:(g + 1) * group_size] - \
                                                             fp_zero_point[:, g].unsqueeze(1)) * scale[:, g].unsqueeze(1)
    return fp_weight.transpose(1, 0)


def test_int4(M, K, N):
    import time

    print("M: {} K: {} N: {}".format(M, K, N))
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b)
    for _ in range(10):
        triton_output = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point)
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        triton_output = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point)
    torch.cuda.synchronize()
    t2 = time.time()
    triton_time = t2 - t1
    print("Triton time cost", (t2 - t1))
    for _ in range(10):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    iters = 512
    t1 = time.time()
    for _ in range(iters):
        torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    t2 = time.time()
    torch_time = t2 - t1
    print("Torch time cost", (t2 - t1))
    return triton_time, torch_time


def test_correct_int4_s1(M=32, K=4096, N=4096):
    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b, group_size=group_size)
    cos = torch.nn.CosineSimilarity(0)
    fp_weight = dequantize_int4(int_b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    print("Quantize cos", cos(fp_weight.flatten().to(torch.float32), b.flatten().to(torch.float32)))
    triton_output = matmul_dequantize_int4_s1(a, int_b, b_scale, b_zero_point, group_size)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print("Output cos", cos(triton_output.flatten().to(torch.float32), torch_output.flatten().to(torch.float32)))


def test_correct_int4_s2(M=32, K=4096, N=4096):
    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b, group_size=group_size)
    cos = torch.nn.CosineSimilarity(0)
    fp_weight = unpack_int4(int_b, b_scale, b_zero_point)
    print("Quantize cos", cos(fp_weight.flatten().to(torch.float32), b.flatten().to(torch.float32)))
    triton_output = matmul_dequantize_int4_s2(a, int_b, b_scale, b_zero_point, group_size)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print("Output cos", cos(triton_output.flatten().to(torch.float32), torch_output.flatten().to(torch.float32)))


def test_correct_int4_gptq(M=32, K=4096, N=4096):
    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point = quantize_int4(b, group_size=group_size)
    cos = torch.nn.CosineSimilarity(0)
    fp_weight = unpack_int4(int_b, b_scale, b_zero_point)
    print("Quantize cos", cos(fp_weight.flatten().to(torch.float32), b.flatten().to(torch.float32)))
    triton_output = matmul_dequantize_int4_gptq(a, int_b, b_scale, b_zero_point, group_size)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print("Output cos", cos(triton_output.flatten().to(torch.float32), torch_output.flatten().to(torch.float32)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # Argument names to use as an x-axis for the plot
        x_vals=[4, 8, 16, 32, 64, 128] + [
            128 * i for i in range(2, 33, 2)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton-s1', 'dequantize', 'triton-s2', 'triton-gptq'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton-s1", "Dequant(GB/s)", "Triton-s2", "Triton-gptq"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('purple', '-'), ('yellow', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, provider):
    K = 4096
    N = 4096
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    if provider == 'triton-s1':
        intb, b_scale, bzp = quantize_int4(b, group_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_dequantize_int4_s1(a, intb, b_scale, bzp, 64), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    if provider == 'triton-s2':
        intb, b_scale, bzp = quantize_int4(b, group_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_dequantize_int4_s2(a, intb, b_scale, bzp, 64), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    if provider == 'dequantize':
        intb, b_scale, bzp = quantize_int4(b, group_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dequantize_int4(intb, b_scale, bzp, 'cuda', torch.float16, 64), quantiles=quantiles)        
        perf = lambda ms: 2 * M * K * 1e-9 / (ms * 1e-3)
    if provider == 'triton-gptq':
        intb, b_scale, bzp = quantize_int4(b, group_size=64)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_dequantize_int4_gptq(a, intb, b_scale, bzp, 64), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def test_model_layer(bs, sqe_len, hidden, inter, tp):
    st1 = 0
    st2 = 0
    t1, t2 = test_int4(bs * sqe_len, hidden, hidden * 3 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, hidden // tp, hidden)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, hidden, inter * 2 // tp)
    st1 += t1
    st2 += t2
    t1, t2 = test_int4(bs * sqe_len, inter // tp, hidden)
    st1 += t1
    st2 += t2
    print("Triton time {} Torch time {}".format(st1, st2))


if __name__ == "__main__":
    # test_correct_int4_s1()
    # test_correct_int4_s2()
    # test_correct_int4_gptq()
    benchmark.run(show_plots=True, print_data=True)
    exit()
    bs = 32
    hidden = 4096
    inter = 11008
    prefill_len = 512
    decode_len = 1
    tp = 1
    test_model_layer(bs, prefill_len, hidden, inter, tp)
    test_model_layer(bs, decode_len, hidden, inter, tp)
