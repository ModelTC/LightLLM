import torch
from typing import Optional
from .base_weight import BaseWeightTpl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.basemodel.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.layernorm import layernorm_forward
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
        assert input.ndim in [2, 3] and self.weight.ndim == 1
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
        if self.weight_name in weights and self.weight is None:
            self.weight = weights[self.weight_name].to(self.data_type_).cuda(get_current_device_id())
        if self.bias_name in weights and self.bias is None:
            self.bias = weights[self.bias_name].to(self.data_type_).cuda(get_current_device_id())


class NoTpGEMMANormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        assert self.bias_name is None
        self.tp_world_size_ = 1
        self.tp_rank_ = 0

    def load_hf_weights(self, weights):
        if self.weight_name in weights and self.weight is None:
            self.weight = (weights[self.weight_name] + 1).to(self.data_type_).cuda(get_current_device_id())


class TpVitPadNormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, head_num: int, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        self.head_num = head_num

    def _pad_tensor_param(self, weight: torch.Tensor):
        assert weight.ndim == 1
        hidden_size = weight.shape[0]
        head_dim = hidden_size // self.head_num
        assert hidden_size % self.head_num == 0

        if self.head_num % self.tp_world_size_ == 0:
            return weight
        else:
            logger.warning(f"padding {self.weight_name} weights in TpVitPadNormWeight")
            pad_head_num = self.tp_world_size_ - (self.head_num % self.tp_world_size_)
            pad_dims = pad_head_num * head_dim
            weight = torch.nn.functional.pad(weight, (0, pad_dims), mode="constant", value=0.0)
            return weight

    def load_hf_weights(self, weights):
        if self.weight_name in weights and self.weight is None:
            t_weight = weights[self.weight_name]
            t_weight = self._pad_tensor_param(t_weight)
            new_hidden_size = t_weight.shape[0]
            split_n_embed = new_hidden_size // self.tp_world_size_
            assert new_hidden_size % self.tp_world_size_ == 0

            start = split_n_embed * self.tp_rank_
            end = split_n_embed * (self.tp_rank_ + 1)

            self.weight = t_weight[start:end].to(self.data_type_).cuda(get_current_device_id())

        if self.bias_name in weights and self.bias is None:
            t_bias = weights[self.bias_name]
            t_bias = self._pad_tensor_param(t_bias)
            new_hidden_size = t_bias.shape[0]
            split_n_embed = new_hidden_size // self.tp_world_size_
            assert new_hidden_size % self.tp_world_size_ == 0

            start = split_n_embed * self.tp_rank_
            end = split_n_embed * (self.tp_rank_ + 1)

            self.bias = t_bias[start:end].to(self.data_type_).cuda(get_current_device_id())


class TpHeadNormWeight(_NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)

    def load_hf_weights(self, weights):
        if self.weight_name in weights and self.weight is None:
            t_weight = weights[self.weight_name]
            start_head_index, end_head_index = self._get_head_tp_split_params(weight=t_weight)
            self.weight: torch.Tensor = (
                t_weight[start_head_index:end_head_index].to(self.data_type_).cuda(get_current_device_id())
            )
            assert self.weight.ndim == 2

        if self.bias_name in weights and self.bias is None:
            t_bias = weights[self.bias_name]
            start_head_index, end_head_index = self._get_head_tp_split_params(weight=t_bias)
            self.bias: torch.Tensor = (
                t_bias[start_head_index:end_head_index].to(self.data_type_).cuda(get_current_device_id())
            )
            assert self.bias.ndim == 2
