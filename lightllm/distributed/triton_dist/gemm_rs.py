import torch
from typing import Optional, Any
from dataclasses import dataclass, field
from lightllm.utils.dist_utils import (
    get_current_rank_in_dp,
    get_dp_world_size,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.device_utils import is_ge_sm90
from lightllm.distributed.overlap_ctx import OverlapCTX
import triton

try:
    from triton_dist.kernels.nvidia.gemm_reduce_scatter import (
        GEMMReduceScatterTensorParallelContext,
        create_gemm_rs_context,
        gemm_rs_producer_persistent,
        gemm_rs_producer_non_persistent,
        update_triton_config,
    )
    from triton_dist.kernels.nvidia.reduce_scatter import (
        reduce_scatter_2d_op,
        ring_reduce,
    )
    from triton_dist.utils import nvshmem_barrier_all_on_stream
except ImportError:
    pass


class TritonDistGemmRSCTX(OverlapCTX):
    def __init__(self, network_config_: dict, dtype: torch.dtype):
        super().__init__(network_config_, dtype)
        args = get_env_start_args()
        self.batch_max_tokens = args.batch_max_tokens
        self.world_size = get_dp_world_size()
        self.rank = get_current_rank_in_dp()
        self.local_world_size = self.world_size // args.nnodes
        self.rs_stream = None
        self._create_context()

    def _create_context(self):
        N = self.network_config_["hidden_size"]
        if self.rs_stream is None:
            self.rs_stream = torch.cuda.Stream()
        self.is_persistent = is_ge_sm90()
        # The default value is True, which means the scatter and gather will be fused into a single kernel.
        self.fuse_scatter = True
        self.gemm_rs_ctx = create_gemm_rs_context(
            max_M=self.batch_max_tokens,
            N=N,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.local_world_size,
            output_dtype=self.dtype,
            rs_stream=self.rs_stream,
        )
        return

    def forward(self, input: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None):
        return gemm_rs(
            a=input,
            b=weight,
            ctx=self.gemm_rs_ctx,
            persistent=self.is_persistent,
            fuse_scatter=self.fuse_scatter,
            output=out,
        )

    def can_run(self, input: torch.Tensor) -> bool:
        return input.shape[0] % self.world_size == 0

    def finalize(self):
        self.gemm_rs_ctx.finalize()


def gemm_rs_op(
    input,
    weight,
    ctx: GEMMReduceScatterTensorParallelContext,
    persistent: bool = True,
    fuse_scatter: bool = False,
    output: Optional[torch.Tensor] = None,
):
    weight = weight.t()
    if fuse_scatter:
        assert ctx.rs_ctx.nnodes == 1, "`fuse_scatter` does not support multi node`"
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    output_dtype = ctx.output_dtype
    num_gemm_sms = ctx.num_gemm_sms

    M, local_K = input.shape
    N = weight.shape[0]
    assert N == ctx.rs_ctx.N

    assert M % world_size == 0
    assert weight.shape[1] == local_K
    M_per_rank = M // world_size
    current_stream = torch.cuda.current_stream()
    rs_stream.wait_stream(current_stream)
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=output_dtype, device=input.device)
    workspace = torch.zeros((world_size,), dtype=torch.int32, device=input.device)
    gemm_out = ctx.get_gemm_out_buf(input)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    if persistent:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M,
                "BLOCK_SIZE_N": ctx.BLOCK_N,
                "BLOCK_SIZE_K": ctx.BLOCK_K,
                "GROUP_SIZE_M": ctx.GROUP_M,
                "NUM_SMS": num_gemm_sms,
                "EPILOGUE_SUBTILE": False,
            },
            num_stages=ctx.stages,
            num_warps=8,
        )
        gemm_rs_producer_persistent(
            input,
            weight,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            fuse_scatter,
            num_gemm_sms,
            triton_config,
        )
    else:
        triton_config = triton.Config(
            {
                "BLOCK_SIZE_M": ctx.BLOCK_M,
                "BLOCK_SIZE_N": ctx.BLOCK_N,
                "BLOCK_SIZE_K": ctx.BLOCK_K,
                "GROUP_SIZE_M": ctx.GROUP_M,
            },
            num_stages=ctx.stages,
            num_warps=8,
        )
        triton_config = update_triton_config(M, N, local_K, input.dtype, world_size, local_world_size, triton_config)
        gemm_rs_producer_non_persistent(
            input,
            weight,
            gemm_out,
            scatter_signal,
            workspace,
            world_size,
            local_world_size,
            fuse_scatter,
            triton_config,
        )

    if not fuse_scatter:
        with torch.cuda.stream(rs_stream):
            # don't allocate memory on other stream: error-prune
            reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output)
        current_stream.wait_stream(rs_stream)
    else:
        nvshmem_barrier_all_on_stream(current_stream)
        ring_reduce(gemm_out, output, ctx.rs_ctx.local_rank, local_world_size)
        nvshmem_barrier_all_on_stream(current_stream)
    return output


def gemm_rs(a, b, ctx, persistent=True, fuse_scatter=False, output: Optional[torch.Tensor] = None):
    """GEMM Reduce-Scatter for Multi-Node

    computes local GEMM (a x b) to generate partial results, followed by `reduce_scatter` to produce c

    Args:
        a (torch.Tensor<bfloat16/float16>): local matmul A matrix. shape: [M, local_K]
        b (torch.Tensor<bfloat16/float16>): local matmul B matrix. shape: [N, local_K]
        ctx(GEMMReduceScatterTensorParallelContext): context

    Returns:
        c (torch.Tensor<bfloat16/float16>): local matmul C matrix. shape: [M // world_size, N]
    """
    c = gemm_rs_op(a, b, ctx, persistent, fuse_scatter, output)
    return c
