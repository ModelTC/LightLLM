import torch
from typing import Optional, Dict
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_dp, get_dp_world_size
from lightllm.common.basemodel.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.layernorm import layernorm_forward
from lightllm.utils.log_utils import init_logger
from .platform_op import PlatformAwareOp

logger = init_logger(__name__)


class RMSNormWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype, bias_name: str = None):
        super().__init__()
        self.dim = dim
        self.weight_name = weight_name
        self.data_type_ = data_type
        self._create_weight()

    def _create_weight(self):
        self.weight: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        )

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])

    def _native_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert input.shape[-1] == self.dim, f"Expected hidden_size to be {self.dim}, but found: {input.shape[-1]}"
        x = input.to(torch.float32)
        x_var = x
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = (x * self.weight).to(self.data_type_)
        if out is None:
            out.copy_(x)
            return out
        return x

    def _cuda_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return rmsnorm_forward(x=input, weight=self.weight, eps=eps, out=out)

    def apply(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(input=input, eps=eps, out=out, alloc_func=alloc_func)


class LayerNormWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype, bias_name: str = None):
        super().__init__()
        self.dim = dim
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self._create_weight()

    def _create_weight(self):
        self.weight: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        )
        self.bias: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        )

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])
        if self.bias_name in weights:
            self.bias.copy_(weights[self.bias_name])

    def _native_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert input.shape[-1] == self.dim, f"Expected hidden_size to be {self.dim}, but found: {input.shape[-1]}"
        x = torch.nn.functional.layer_norm(
            input, normalized_shape=[self.dim], weight=self.weight, bias=self.bias, eps=eps
        )
        if out is None:
            out.copy_(x.to(self.data_type_))
            return out
        return x.to(self.data_type_)

    def _cuda_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return layernorm_forward(x=input, weight=self.weight, bias=self.bias, eps=eps, out=out)

    def apply(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(input=input, eps=eps, out=out, alloc_func=alloc_func)


class TpRMSNormWeight(RMSNormWeight):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype, bias_name: str = None):
        super().__init__(dim=dim, weight_name=weight_name, data_type=data_type, bias_name=bias_name)
        self.tp_world_size_ = get_dp_world_size()
        self.tp_rank_ = get_current_rank_in_dp()
        self.dim = self._get_tp_padded_dim(dim=dim)
        self.repeat_times_ = 1

    def _get_tp_padded_dim(self, dim: int):
        """
        Get the padded dimension for the weight.
        1. if dim is divisible by tp_world_size_, return dim
        2. if dim is greater than tp_world_size_, return (dim + tp_world_size_ - 1) // tp_world_size_ * tp_world_size_
        3. if dim is less than tp_world_size_, assert tp_world_size_ is divisible by dim, and return dim
        """
        if dim % self.tp_world_size_ == 0:
            return dim // self.tp_world_size_

        if dim > self.tp_world_size_:
            return (dim + self.tp_world_size_ - 1) // self.tp_world_size_ * self.tp_world_size_
        else:
            assert (
                self.tp_world_size_ % dim == 0
            ), f"tp_world_size_ must be divisible by dim, but found: {self.tp_world_size_} % {dim}"
            self.repeat_times_ = self.tp_world_size_ // dim
            return dim * self.repeat_times_ // self.tp_world_size_

    def load_hf_weights(self, weights):
        if self.weight_name in weights and self.weight is None:
            t_weight = weights[self.weight_name]
            hidden_size = t_weight.shape[0]
            split_hidden_size = hidden_size // self.tp_world_size_

            start = split_hidden_size * self.tp_rank_ // self.repeat_times_
            end = min(split_hidden_size * (self.tp_rank_ + 1) // self.repeat_times_, hidden_size)

            self.weight[:, end - start].copy_(t_weight[start:end].to(self.data_type_))
            # the padding part is zero
            self.weight[:, end:].zero_()
