import torch
from typing import Dict, Optional, Tuple
from .base_weight import BaseWeightTpl


class ParameterWeight(BaseWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
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
        # Create weights if shapes are provided
        if weight_shape is not None:
            self._create_weight()

    def _create_weight(self):
        """Create weight and bias tensors with pre-allocated memory."""
        if self.weight_shape is not None:
            self.weight = torch.empty(*self.weight_shape, dtype=self.data_type_, device=self.device_id_)
        if self.bias_name is not None and self.bias_shape is not None:
            self.bias = torch.empty(*self.bias_shape, dtype=self.data_type_, device=self.device_id_)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_name in weights:
            t_weight = weights[self.weight_name]
            if self.weight is None:
                # If weight was not pre-created, create it now based on loaded shape
                self.weight = torch.empty(*t_weight.shape, dtype=self.data_type_, device=self.device_id_)
            self.weight.copy_(t_weight.to(self.data_type_))
        if self.bias_name is not None and self.bias_name in weights:
            t_bias = weights[self.bias_name]
            if self.bias is None:
                # If bias was not pre-created, create it now based on loaded shape
                self.bias = torch.empty(*t_bias.shape, dtype=self.data_type_, device=self.device_id_)
            self.bias.copy_(t_bias.to(self.data_type_))

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


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
        super().__init__(weight_name, data_type, bias_name, tp_weight_shape, tp_bias_shape)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)

        if self.weight_name in weights:
            t_weight = weights[self.weight_name][start:end]
            if self.weight is None:
                # If weight was not pre-created, create it now based on loaded shape
                self.weight = torch.empty(*t_weight.shape, dtype=self.data_type_, device=self.device_id_)
            self.weight.copy_(t_weight.to(self.data_type_))
        if self.bias_name is not None and self.bias_name in weights:
            t_bias = weights[self.bias_name][start:end]
            if self.bias is None:
                # If bias was not pre-created, create it now based on loaded shape
                self.bias = torch.empty(*t_bias.shape, dtype=self.data_type_, device=self.device_id_)
            self.bias.copy_(t_bias.to(self.data_type_))
