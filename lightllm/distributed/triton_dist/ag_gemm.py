import torch

# try:
#     from triton_dist.kernels.nvidia.allgather_gemm import (
#         AllGatherGEMMTensorParallelContext,
#         rowise_ag_gemm_dispatcher,
#     )
#     from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
#     from triton_dist.kernels.nvidia.allgather_gemm import (
#         AllGatherGEMMTensorParallelContext,
#         get_auto_all_gather_method,
#         ag_gemm,
#     )
#     from triton_dist.utils import nvshmem_barrier_all_on_stream
# except ImportError:
#     AllGatherGEMMTensorParallelContext = None
