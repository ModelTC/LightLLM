import torch
from typing import Optional, Dict
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_dp, get_dp_world_size
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.layernorm import layernorm_forward
from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward
from .platform_op import PlatformAwareOp


class RMSNormWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype, **kwargs):
        super().__init__()
        self.dim = dim
        self.weight_name = weight_name
        self.data_type_ = data_type
        self._create_weight()

    def _create_weight(self):
        self.weight: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        self.weight.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])
            self.weight.load_ok = True

    def verify_load(self):
        return self.weight.load_ok

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
        if out is not None:
            out.copy_(x)
            return out
        return x

    def _triton_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert (
            input.ndim in [2, 3] and self.weight.ndim == 1
        ), f"input.ndim: {input.ndim} != 2 or weight.ndim: {self.weight.ndim} != 1"
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return rmsnorm_forward(x=input, weight=self.weight, eps=eps, out=out)

    def _cuda_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        # only triton implementation is supported for rmsnorm on cuda platform
        return self._triton_forward(input=input, eps=eps, out=out, alloc_func=alloc_func)

    def _musa_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        # triton implementation is supported by musa.
        return self._triton_forward(input=input, eps=eps, out=out, alloc_func=alloc_func)

    def __call__(
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
        self.weight: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        self.bias: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        self.weight.load_ok = False
        self.bias.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])
            self.weight.load_ok = True
        if self.bias_name in weights:
            self.bias.copy_(weights[self.bias_name])
            self.bias.load_ok = True

    def verify_load(self):
        return self.weight.load_ok and self.bias.load_ok

    def _native_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert input.shape[-1] == self.dim, f"Expected hidden_size to be {self.dim}, but found: {input.shape[-1]}"
        x = torch.nn.functional.layer_norm(
            input, normalized_shape=[self.dim], weight=self.weight, bias=self.bias, eps=eps
        )
        if out is not None:
            out.copy_(x.to(self.data_type_))
            return out
        return x.to(self.data_type_)

    def _triton_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        # assert input.ndim == 2 and self.weight.ndim == 1
        print(input.shape)
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        out[:] = layernorm_forward(x=input, weight=self.weight, bias=self.bias, eps=eps)
        return out

    def _cuda_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        # only triton implementation is supported for layernorm on cuda platform
        return self._triton_forward(input=input, eps=eps, out=out, alloc_func=alloc_func)

    def _musa_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        # triton implementation is supported by musa.
        return self._triton_forward(input=input, eps=eps, out=out, alloc_func=alloc_func)

    def __call__(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(input=input, eps=eps, out=out, alloc_func=alloc_func)


class TpRMSNormWeight(RMSNormWeight):
    def __init__(self, head_num, head_dim, weight_name: str, data_type: torch.dtype):
        padded_head_num = self._get_tp_padded_head_num(head_num)
        dim = padded_head_num * head_dim
        super().__init__(dim=dim, weight_name=weight_name, data_type=data_type)
        self.repeat_times_ = 1

    def _get_tp_padded_head_num(self, head_num: int):
        """
        Get the padded dimension for the weight.
        1. If head_num is divisible by tp_world_size_, return head_num.
        2. If head_num is greater than tp_world_size_, return:
           (head_num + tp_world_size_ - 1) // tp_world_size_ * tp_world_size_
        3. If head_num is less than tp_world_size_, assert tp_world_size_ is
           divisible by head_num, and return head_num.
        """
        self.tp_world_size_ = get_dp_world_size()
        if head_num % self.tp_world_size_ == 0:
            return head_num // self.tp_world_size_

        if head_num > self.tp_world_size_:
            return (head_num + self.tp_world_size_ - 1) // self.tp_world_size_ * self.tp_world_size_
        else:
            assert (
                self.tp_world_size_ % head_num == 0
            ), f"tp_world_size_ must be divisible by head_num, but found: {self.tp_world_size_} % {head_num}"
            self.repeat_times_ = self.tp_world_size_ // head_num
            return head_num * self.repeat_times_ // self.tp_world_size_

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            t_weight = weights[self.weight_name]
            hidden_size = t_weight.shape[0]
            split_hidden_size = hidden_size // self.tp_world_size_

            start = split_hidden_size * self.tp_rank_ // self.repeat_times_
            end = min(split_hidden_size * (self.tp_rank_ + 1) // self.repeat_times_, hidden_size)

            self.weight[: end - start].copy_(t_weight[start:end].to(self.data_type_))
            # the padding part is zero
            self.weight[end - start :].zero_()
            self.weight.load_ok = True


class NoTpGEMMANormWeight(RMSNormWeight):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype):
        super().__init__(dim=dim, weight_name=weight_name, data_type=data_type)
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            self.weight.copy_(weights[self.weight_name])
            self.weight += 1


class QKRMSNORMWeight(RMSNormWeight):
    def __init__(self, dim: int, weight_name: str, data_type: torch.dtype):
        super().__init__(dim=dim, weight_name=weight_name, data_type=data_type)
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def _native_forward(
        self,
        input: torch.Tensor,
        eps: float,
    ) -> None:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert input.shape[-1] == self.dim, f"Expected hidden_size to be {self.dim}, but found: {input.shape[-1]}"
        head_dim = self.weight.shape[0]
        x = input.to(torch.float32)
        x = x.view(-1, head_dim)
        x_var = x
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = (x * self.weight).to(self.data_type_)
        x = x.view(-1, input.shape[-1])
        input.copy_(x)
        return

    def _triton_forward(self, input: torch.Tensor, eps: float) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        return qk_rmsnorm_forward(x=input, weight=self.weight, eps=eps)

    def _cuda_forward(
        self,
        input: torch.Tensor,
        eps: float,
    ) -> None:
        self._triton_forward(input=input, eps=eps)
        return

    def _musa_forward(self, input: torch.Tensor, eps: float) -> torch.Tensor:
        # musa implementation is supported by musa triton on musa platform
        return self._triton_forward(input=input, eps=eps)

    def __call__(
        self,
        input: torch.Tensor,
        eps: float,
    ) -> None:
        return self._forward(input=input, eps=eps)
