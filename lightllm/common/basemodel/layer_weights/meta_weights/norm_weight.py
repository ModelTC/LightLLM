import torch
from typing import Optional
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.layernorm import layernorm_forward
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class NormWeight(BaseWeightTpl):
    def __init__(self, norm_dim: int, weight_name, data_type, bias_name=None):
        super().__init__()
        self.norm_dim = norm_dim
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self.weight = None
        self.bias = None
        self.is_weight_ready = False
        self.is_bias_ready = False
        self._create_weight()

    def _create_weight(self):
        device = f"cuda:{get_current_device_id()}"
        self.weight = torch.empty(self.norm_dim, dtype=self.data_type_, device=device)
        self.bias = (
            torch.empty(self.norm_dim, dtype=self.data_type_, device=device) if self.bias_name is not None else None
        )

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])
            self.is_weight_ready = True
        if self.bias_name in weights:
            self.bias.copy_(weights[self.bias_name])
            self.is_bias_ready = True

    def verify_load(self):
        return self.is_weight_ready and (self.bias_name is None or self.is_bias_ready)

    def rmsnorm_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim in [2, 3] and self.weight.ndim == 1
        assert self.bias is None
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return rmsnorm_forward(x=input, weight=self.weight, eps=eps, out=out)


class GEMMANormWeight(NormWeight):
    def __init__(self, norm_dim: int, weight_name, data_type, bias_name=None):
        super().__init__(norm_dim, weight_name, data_type, bias_name)

    def load_hf_weights(self, weights):
        # TODO: 这里直接 +1 会不会导致精度问题? 计算时要求 (1.0 + weight.float()) ?
        if self.weight_name in weights:
            self.weight.copy_((weights[self.weight_name] + 1).to(self.data_type_))
            self.is_weight_ready = True


class TpNormWeight(NormWeight):
    def __init__(self, norm_dim: int, weight_name, data_type, bias_name=None):
        super().__init__(norm_dim, weight_name, data_type, bias_name)

    def load_hf_weights(self, weights):
        start = self.norm_dim * self.tp_rank_
        end = self.norm_dim * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name][start:end].to(self.data_type_))
            self.is_weight_ready = True
        if self.bias_name in weights:
            self.bias.copy_(weights[self.bias_name][start:end].to(self.data_type_))
            self.is_bias_ready = True
