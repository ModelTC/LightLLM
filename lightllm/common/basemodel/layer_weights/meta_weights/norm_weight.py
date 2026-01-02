import torch
from typing import Optional
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.layernorm import layernorm_forward


class _NormWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__()
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self.weight: torch.Tensor = None
        self.bias: Optional[torch.Tensor] = None

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok

    def rmsnorm_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert self.bias is None
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return rmsnorm_forward(x=input, weight=self.weight, eps=eps, out=out)

    def layernorm_forward(
        self, input: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        assert input.ndim == 2 and self.weight.ndim == 1
        assert self.bias is not None

        _tout = layernorm_forward(x=input, weight=self.weight, bias=self.bias, eps=eps)
        if out is None:
            return _tout
        else:
            out.copy_(_tout)
            return out


class NoTpNormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name=weight_name, data_type=data_type, bias_name=bias_name)
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight = weights[self.weight_name].to(self.data_type_).cuda(get_current_device_id())
        if self.bias_name in weights:
            self.bias = weights[self.bias_name].to(self.data_type_).cuda(get_current_device_id())


class NoTpGEMMANormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        assert self.bias_name is None
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight = (weights[self.weight_name] + 1).to(self.data_type_).cuda(get_current_device_id())


class TpNormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        self.split_n_embed = split_n_embed

    def load_hf_weights(self, weights):
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight = weights[self.weight_name][start:end].to(self.data_type_).cuda(get_current_device_id())
        if self.bias_name in weights:
            self.bias = weights[self.bias_name][start:end].to(self.data_type_).cuda(get_current_device_id())


class TpHeadNormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, tp_head_num, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        self.tp_head_num = tp_head_num
        assert self.tp_head_num > 0

    def load_hf_weights(self, weights):
        start = self.tp_head_num * self.tp_rank_
        end = self.tp_head_num * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            self.weight: torch.Tensor = (
                weights[self.weight_name][start:end].to(self.data_type_).cuda(get_current_device_id())
            )
            assert self.weight.ndim == 2
        if self.bias_name in weights:
            self.bias: torch.Tensor = (
                weights[self.bias_name][start:end].to(self.data_type_).cuda(get_current_device_id())
            )
            assert self.bias.ndim == 2
