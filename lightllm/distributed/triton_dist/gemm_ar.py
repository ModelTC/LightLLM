import torch
import triton
from dataclasses import dataclass
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp
from lightllm.utils.device_utils import is_ge_sm90
from lightllm.distributed.overlap_ctx import OverlapCTX
from lightllm.utils.device_utils import get_device_sm_count
from typing import Optional

try:
    from triton_dist.kernels.nvidia.gemm_allreduce import (
        GemmARContext,
        LLGemmARContext,
    )
    from triton_dist.utils import nvshmem_create_tensor, launch_cooperative_grid_options
    from triton_dist.utils import nvshmem_barrier_all_on_stream, is_nvshmem_multimem_supported
    from triton_dist.kernels.nvidia.gemm_allreduce import (
        persistent_gemm_notify,
        consumer_all_reduce,
        kernel_fused_gemm_allreduce,
    )


except ImportError:
    pass


class TritonDistGemmARCTX(OverlapCTX):
    def __init__(self, network_config_: dict, dtype: torch.dtype):
        super().__init__(network_config_, dtype)
        args = get_env_start_args()
        self.batch_max_tokens = args.batch_max_tokens
        self.world_size = get_dp_world_size()
        self.rank = get_current_rank_in_dp()
        self.local_world_size = self.world_size // args.nnodes
        self.ar_stream = None
        self.copy_to_local = False
        self.USE_MULTIMEM_ST = is_nvshmem_multimem_supported()
        assert self.USE_MULTIMEM_ST, "The multimem-st is required for gemm-ar"
        self._create_context()

    def _create_context(self):
        N = self.network_config_["hidden_size"]
        if self.ar_stream is None:
            self.ar_stream = torch.cuda.Stream()
        self.is_persistent = is_ge_sm90()
        self.N = N
        self.P_NUM_COMM_SMS = 4
        self.D_NUM_COMM_SMS = 16
        self.P_NUM_GEMM_SMS = get_device_sm_count() - self.P_NUM_COMM_SMS
        self.D_NUM_GEMM_SMS = get_device_sm_count() - self.D_NUM_COMM_SMS

        # TODO: need to be optimized
        self.ll_gemm_config = triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1,
                "NUM_GEMM_SMS": self.D_NUM_GEMM_SMS,
            },
            num_stages=5,
            num_warps=4,
        )
        self.gemm_config = triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_GEMM_SMS": self.P_NUM_GEMM_SMS,
            },
            num_stages=4,
            num_warps=8,
        )

        self.gemm_ar_ctx = create_gemm_ar_context(
            ar_stream=self.ar_stream,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.local_world_size,
            max_M=self.batch_max_tokens,
            N=N,
            dtype=self.dtype,
            NUM_COMM_SMS=self.P_NUM_COMM_SMS,
        )
        self.ll_gemm_ar_ctx = create_ll_gemm_ar_context(
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=self.local_world_size,
            max_M=self.batch_max_tokens,
            N=N,
            dtype=self.dtype,
            NUM_COMM_SMS=self.D_NUM_COMM_SMS,
        )
        return

    def forward(self, input: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None):
        weight = weight.t()
        assert input.shape[0] <= self.batch_max_tokens and weight.shape[0] == self.N
        #  batchsize <= 128 use low latency kernel
        if input.shape[0] <= 128:
            ar_out = low_latency_gemm_allreduce_op(
                self.ll_gemm_ar_ctx,
                input,
                weight,
                self.ll_gemm_config,
                copy_to_local=self.copy_to_local,
                USE_MULTIMEM_ST=self.USE_MULTIMEM_ST,
                output=out,
            )
        else:  # batchsize > 128 use gemm kernel
            ar_out = gemm_allreduce_op(
                self.gemm_ar_ctx,
                input,
                weight,
                self.gemm_config,
                copy_to_local=self.copy_to_local,
                USE_MULTIMEM_ST=self.USE_MULTIMEM_ST,
                output=out,
            )
        return ar_out

    def can_run(self, input: torch.Tensor) -> bool:
        return input.shape[0] <= self.batch_max_tokens

    def finalize(self):
        self.gemm_ar_ctx.finalize()
        self.ll_gemm_ar_ctx.finalize()


def create_gemm_ar_context(
    ar_stream: torch.cuda.Stream,
    rank,
    world_size,
    local_world_size,
    max_M,
    N,
    dtype,
    MIN_BLOCK_SIZE_M=16,
    MIN_BLOCK_SIZE_N=16,
    NUM_COMM_SMS=132,
):
    assert local_world_size == world_size
    gemm_out_buf = nvshmem_create_tensor((max_M, N), dtype)
    symm_ar_out_buf = nvshmem_create_tensor((max_M, N), dtype)
    gemm_barrier_buf = nvshmem_create_tensor(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)), torch.int32
    )
    multi_st_barrier_buf = nvshmem_create_tensor((world_size * NUM_COMM_SMS,), torch.int32)
    grid_barrier_buf = torch.zeros((1,), dtype=torch.int32, device=torch.cuda.current_device())
    gemm_barrier_buf.zero_()
    multi_st_barrier_buf.zero_()
    nvshmem_barrier_all_on_stream()
    return GemmARContext(
        symm_gemm_out_buf=gemm_out_buf,
        symm_ar_out_buf=symm_ar_out_buf,
        gemm_barrier_buf=gemm_barrier_buf,
        multi_st_barrier_buf=multi_st_barrier_buf,
        grid_barrier_buf=grid_barrier_buf,
        NUM_COMM_SMS=NUM_COMM_SMS,
        ar_stream=ar_stream,
    )


def create_ll_gemm_ar_context(
    rank,
    world_size,
    local_world_size,
    max_M,
    N,
    dtype,
    MIN_BLOCK_SIZE_M=16,
    MIN_BLOCK_SIZE_N=16,
    NUM_COMM_SMS=132,
    num_phases=2,
):
    ar_stream = torch.cuda.Stream(priority=-1)
    ctxs = []
    for i in range(num_phases):
        ctxs.append(
            create_gemm_ar_context(
                ar_stream, rank, world_size, local_world_size, max_M, N, dtype, NUM_COMM_SMS=NUM_COMM_SMS
            )
        )
    nvshmem_barrier_all_on_stream()
    return LLGemmARContext(ctxs=ctxs, num_phases=num_phases, phase=0)


def low_latency_gemm_allreduce_op(
    ctx: LLGemmARContext,
    a,
    b,
    gemm_config: triton.Config,
    copy_to_local=True,
    USE_MULTIMEM_ST=True,
    output: torch.Tensor = None,
):
    ctx.update_phase()
    M, N = a.shape[0], b.shape[0]
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    grid_barrier = ctx.grid_barrier_buf

    NUM_COMM_SMS = ctx.NUM_COMM_SMS

    if output is None:
        ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    else:
        ar_out = output
    grid = lambda META: (
        NUM_COMM_SMS
        + min(META["NUM_GEMM_SMS"], triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )
    kernel_fused_gemm_allreduce[grid](
        a,
        b,
        symm_c,
        symm_ar_out,
        ar_out,  #
        gemm_barrier,
        multi_st_barrier,
        grid_barrier,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        symm_c.stride(0),
        symm_c.stride(1),  #
        **gemm_config.all_kwargs(),  #
        NUM_COMM_SMS=NUM_COMM_SMS,
        USE_MULTIMEM_ST=USE_MULTIMEM_ST,
        FUSE_OUTPUT_CP=copy_to_local,
        use_cooperative=True,
        **launch_cooperative_grid_options()
    )
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[: M * N].reshape(M, N)
    return ar_out


def gemm_allreduce_op(
    ctx: GemmARContext,
    a,
    b,
    gemm_config: triton.Config,
    copy_to_local=True,
    USE_MULTIMEM_ST=True,
    output: torch.Tensor = None,
):

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)

    M, N = a.shape[0], b.shape[0]

    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ar_out = ctx.symm_ar_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    if output is None:
        ar_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    else:
        ar_out = output
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]
    # add mask in `consumer_all_reduce` can remove this constraint
    assert N % BLOCK_SIZE_N == 0
    persistent_gemm_notify(a, b, symm_c, gemm_barrier, gemm_config)
    with torch.cuda.stream(ar_stream):
        consumer_all_reduce(
            symm_c,
            symm_ar_out,
            ar_out,
            gemm_barrier,
            multi_st_barrier,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_COMM_SMS=NUM_COMM_SMS,
            USE_MULTIMEM_ST=USE_MULTIMEM_ST,
        )
    current_stream.wait_stream(ar_stream)
    # out still in comm buffer, copy to user buffer
    if USE_MULTIMEM_ST and copy_to_local:
        ar_out.copy_(symm_ar_out.reshape(-1)[: M * N].reshape(M, N))
    # some ranks may not reset the barrier, other ranks will read dirty data during allreduce in the next iter.
    nvshmem_barrier_all_on_stream(current_stream)
    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ar_out.reshape(-1)[: M * N].reshape(M, N)
    return ar_out
