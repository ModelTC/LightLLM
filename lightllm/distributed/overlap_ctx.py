import torch
from typing import Optional
from dataclasses import dataclass
from lightllm.utils.envs_utils import get_env_start_args


class OverlapCTX:
    def __init__(self, network_config_: dict, dtype: torch.dtype):
        self.network_config_ = network_config_
        self.dtype = dtype

    def _create_context(self):
        raise NotImplementedError("need to impl")

    def forward(self, input: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("need to impl")

    def finalize(self):
        raise NotImplementedError("need to impl")


@dataclass
class OverlapWrapper:
    """
    这里完全根据transformers的推理特征来设计折叠对象。例如gemm_rs，gemm_ar只能给COLMM使用。
    """

    row_ctx: Optional[OverlapCTX] = None
    col_ctx: Optional[OverlapCTX] = None

    def is_empty(self) -> bool:
        return self.row_ctx is None and self.col_ctx is None


def create_overlap_ctx(network_config_: dict, dtype: torch.dtype) -> OverlapCTX:
    args = get_env_start_args()
    if args.overlap_type is None:
        return OverlapWrapper(row_ctx=None, col_ctx=None)
    from lightllm.distributed.triton_dist.gemm_rs import TritonDistGemmRSCTX

    if args.overlap_type == "gemm_rs_triton_dist":
        return OverlapWrapper(row_ctx=None, col_ctx=TritonDistGemmRSCTX(network_config_, dtype))
    else:
        raise ValueError(f"Unsupported overlap type: {args.overlap_type}")
