import torch
from typing import Dict, Optional, Tuple
from .base_weight import BaseWeightTpl


class ParameterWeight(BaseWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        weight_shape: Optional[Tuple[int, ...]],
        bias_name: Optional[str] = None,
        bias_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.data_type_ = data_type
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        if weight_shape is not None:
            self._create_weight()

    def _create_weight(self):
        if self.weight_shape is not None:
            self.weight = torch.empty(*self.weight_shape, dtype=self.data_type_, device=self.device_id_)
            self.weight.load_ok = False
        if self.bias_name is not None and self.bias_shape is not None:
            self.bias = torch.empty(*self.bias_shape, dtype=self.data_type_, device=self.device_id_)
            self.bias.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name in weights:
            t_weight = weights[self.weight_name]
            self.weight.copy_(t_weight.to(self.data_type_))
            self.weight.load_ok = True
        if self.bias_name is not None and self.bias_name in weights:
            t_bias = weights[self.bias_name]
            self.bias.copy_(t_bias.to(self.data_type_))
            self.bias.load_ok = True

    def verify_load(self) -> bool:
        if self.weight is not None and not getattr(self.weight, "load_ok", False):
            return False
        if self.bias is not None and not getattr(self.bias, "load_ok", False):
            return False
        return True


class TpParameterWeight(ParameterWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        bias_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.split_n_embed = split_n_embed
        # Calculate TP-split shapes if full shapes are provided
        tp_weight_shape = None
        tp_bias_shape = None
        if weight_shape is not None:
            tp_weight_shape = (split_n_embed,) + weight_shape[1:]
        if bias_shape is not None:
            tp_bias_shape = (split_n_embed,) + bias_shape[1:]
        super().__init__(weight_name, data_type, tp_weight_shape, bias_name, tp_bias_shape)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            t_weight = weights[self.weight_name][start:end]
            self.weight.copy_(t_weight.to(self.data_type_))
            self.weight.load_ok = True
        if self.bias_name is not None and self.bias_name in weights:
            t_bias = weights[self.bias_name][start:end]
            self.bias.copy_(t_bias.to(self.data_type_))
            self.bias.load_ok = True
