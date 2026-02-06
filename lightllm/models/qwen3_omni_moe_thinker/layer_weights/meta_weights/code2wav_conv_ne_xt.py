import math
import torch
import numpy as np
from typing import Dict, Optional
from transformers.activations import ACT2FN
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.basemodel.layer_weights.meta_weights.platform_op import PlatformAwareOp
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp


class Qwen3OmniMoeConvNeXtBlockWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(
        self,
        dim: int,
        dwconv,
        norm_weight_name: str,
        norm_bias_name: str,
        pwconv1_weight_name: str,
        pwconv1_bias_name: str,
        pwconv2_weight_name: str,
        pwconv2_bias_name: str,
        gamma_name: str,
        data_type: torch.dtype,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.dwconv = dwconv
        self.norm_weight_name = norm_weight_name
        self.norm_bias_name = norm_bias_name
        self.pwconv1_weight_name = pwconv1_weight_name
        self.pwconv1_bias_name = pwconv1_bias_name
        self.pwconv2_weight_name = pwconv2_weight_name
        self.pwconv2_bias_name = pwconv2_bias_name
        self.gamma_name = gamma_name
        self.data_type_ = data_type
        self.eps = eps

        self._create_weight()

    def _create_weight(self):
        self.norm_weight: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)
        self.norm_bias: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)

        self.pwconv1_weight: torch.Tensor = torch.empty(
            4 * self.dim, self.dim, dtype=self.data_type_, device=self.device_id_
        )
        self.pwconv1_bias: torch.Tensor = torch.empty(4 * self.dim, dtype=self.data_type_, device=self.device_id_)

        self.pwconv2_weight: torch.Tensor = torch.empty(
            self.dim, 4 * self.dim, dtype=self.data_type_, device=self.device_id_
        )
        self.pwconv2_bias: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)

        self.gamma: torch.Tensor = torch.empty(self.dim, dtype=self.data_type_, device=self.device_id_)

        self.norm_weight.load_ok = False
        self.norm_bias.load_ok = False
        self.pwconv1_weight.load_ok = False
        self.pwconv1_bias.load_ok = False
        self.pwconv2_weight.load_ok = False
        self.pwconv2_bias.load_ok = False
        self.gamma.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.norm_weight_name in weights:
            t = weights[self.norm_weight_name]
            assert t.shape == (self.dim,)
            self.norm_weight.copy_(t.to(self.data_type_))
            self.norm_weight.load_ok = True

        if self.norm_bias_name in weights:
            t = weights[self.norm_bias_name]
            assert t.shape == (self.dim,)
            self.norm_bias.copy_(t.to(self.data_type_))
            self.norm_bias.load_ok = True

        if self.pwconv1_weight_name in weights:
            t = weights[self.pwconv1_weight_name]
            assert t.shape == (4 * self.dim, self.dim)
            self.pwconv1_weight.copy_(t.to(self.data_type_))
            self.pwconv1_weight.load_ok = True

        if self.pwconv1_bias_name in weights:
            t = weights[self.pwconv1_bias_name]
            assert t.shape == (4 * self.dim,)
            self.pwconv1_bias.copy_(t.to(self.data_type_))
            self.pwconv1_bias.load_ok = True

        if self.pwconv2_weight_name in weights:
            t = weights[self.pwconv2_weight_name]
            assert t.shape == (self.dim, 4 * self.dim)
            self.pwconv2_weight.copy_(t.to(self.data_type_))
            self.pwconv2_weight.load_ok = True

        if self.pwconv2_bias_name in weights:
            t = weights[self.pwconv2_bias_name]
            assert t.shape == (self.dim,)
            self.pwconv2_bias.copy_(t.to(self.data_type_))
            self.pwconv2_bias.load_ok = True

        if self.gamma_name in weights:
            t = weights[self.gamma_name]
            assert t.shape == (self.dim,)
            self.gamma.copy_(t.to(self.data_type_))
            self.gamma.load_ok = True

    def verify_load(self):
        return (
            self.norm_weight.load_ok
            and self.norm_bias.load_ok
            and self.pwconv1_weight.load_ok
            and self.pwconv1_bias.load_ok
            and self.pwconv2_weight.load_ok
            and self.pwconv2_bias.load_ok
            and self.gamma.load_ok
        )

    def _native_forward(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)  # [B, C, L]
        hidden_states = hidden_states.permute(0, 2, 1)  # [B, L, C]

        mean = hidden_states.mean(dim=-1, keepdim=True)
        var = (hidden_states - mean).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(var + self.eps)
        hidden_states = hidden_states * self.norm_weight.view(1, 1, -1) + self.norm_bias.view(1, 1, -1)

        hidden_states = torch.nn.functional.linear(hidden_states, self.pwconv1_weight, self.pwconv1_bias)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = torch.nn.functional.linear(hidden_states, self.pwconv2_weight, self.pwconv2_bias)

        hidden_states = hidden_states * self.gamma.view(1, 1, -1)

        hidden_states = hidden_states.permute(0, 2, 1)  # [B, C, L]
        hidden_states = input + hidden_states

        if out is not None:
            out.copy_(hidden_states)
            return out
        return hidden_states

    def _cuda_forward(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        if out is None:
            result = self._native_forward(hidden_states=hidden_states, out=None, _alloc_func=alloc_func)
            return result
        result = self._native_forward(hidden_states=hidden_states, out=None, _alloc_func=alloc_func)
        out.copy_(result)
        return out

    def _musa_forward(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._cuda_forward(hidden_states=hidden_states, out=out, alloc_func=alloc_func)

    def __call__(
        self, hidden_states: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(hidden_states=hidden_states, out=out, alloc_func=alloc_func)
