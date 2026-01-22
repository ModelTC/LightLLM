import torch
from abc import abstractmethod
from lightllm.common.quantization.quantize_method import (
    WeightPack,
    QuantizationMethod,
)
from typing import Optional
from lightllm.utils.dist_utils import (
    get_global_rank,
    get_global_world_size,
)


class FuseMoeBaseImpl:
    def __init__(
        self,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        redundancy_expert_num: int,
        routed_scaling_factor: float,
        quant_method: QuantizationMethod,
    ):
        self.n_routed_experts = n_routed_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.redundancy_expert_num = redundancy_expert_num
        self.routed_scaling_factor = routed_scaling_factor
        self.quant_method = quant_method
        self.global_rank_ = get_global_rank()
        self.global_world_size = get_global_world_size()
        self.total_expert_num_contain_redundancy = (
            self.n_routed_experts + self.redundancy_expert_num * self.global_world_size
        )
        self.workspace = self.create_workspace()

    @abstractmethod
    def create_workspace(self):
        pass

    @abstractmethod
    def __call__(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        correction_bias: Optional[torch.Tensor],
        scoring_func: str,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
    ):
        pass
