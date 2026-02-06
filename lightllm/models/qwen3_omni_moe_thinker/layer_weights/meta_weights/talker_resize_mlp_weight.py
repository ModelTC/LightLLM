import torch
import numpy as np
from typing import Dict, Optional
from transformers.activations import ACT2FN
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.basemodel.layer_weights.meta_weights.platform_op import PlatformAwareOp
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp


class Qwen3OmniMoeTalkerResizeMLPWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(
        self,
        in_dim: int,
        intermediate_dim: int,
        out_dim: int,
        fc1_weight_name: str,
        fc1_bias_name: str,
        fc2_weight_name: str,
        fc2_bias_name: str,
        hidden_act: str,
        data_type: torch.dtype,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.intermediate_dim = intermediate_dim
        self.out_dim = out_dim
        self.fc1_weight_name: str = fc1_weight_name
        self.fc1_bias_name: str = fc1_bias_name
        self.fc2_weight_name: str = fc2_weight_name
        self.fc2_bias_name: str = fc2_bias_name
        self.data_type_ = data_type
        self.act_fn = ACT2FN[hidden_act]
        self._create_weight()

    def _create_weight(self):
        self.fc1_weight: torch.Tensor = torch.empty(
            self.intermediate_dim, self.in_dim, dtype=self.data_type_, device=self.device_id_
        )
        self.fc1_bias: torch.Tensor = torch.empty(self.intermediate_dim, dtype=self.data_type_, device=self.device_id_)
        self.fc2_weight: torch.Tensor = torch.empty(
            self.out_dim, self.intermediate_dim, dtype=self.data_type_, device=self.device_id_
        )
        self.fc2_bias: torch.Tensor = torch.empty(self.out_dim, dtype=self.data_type_, device=self.device_id_)
        self.fc1_weight.load_ok = False
        self.fc1_bias.load_ok = False
        self.fc2_weight.load_ok = False
        self.fc2_bias.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.fc1_weight_name in weights:
            t = weights[self.fc1_weight_name]
            assert t.shape == (self.intermediate_dim, self.in_dim)
            self.fc1_weight.copy_(t.to(self.data_type_))
            self.fc1_weight.load_ok = True
        if self.fc1_bias_name in weights:
            t = weights[self.fc1_bias_name]
            assert t.shape == (self.intermediate_dim,)
            self.fc1_bias.copy_(t.to(self.data_type_))
            self.fc1_bias.load_ok = True
        if self.fc2_weight_name in weights:
            t = weights[self.fc2_weight_name]
            assert t.shape == (self.out_dim, self.intermediate_dim)
            self.fc2_weight.copy_(t.to(self.data_type_))
            self.fc2_weight.load_ok = True
        if self.fc2_bias_name in weights:
            t = weights[self.fc2_bias_name]
            assert t.shape == (self.out_dim,)
            self.fc2_bias.copy_(t.to(self.data_type_))
            self.fc2_bias.load_ok = True

    def verify_load(self):
        return self.fc1_weight.load_ok and self.fc1_bias.load_ok and self.fc2_weight.load_ok and self.fc2_bias.load_ok

    def _native_forward(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        in_dim = hidden_state.shape[-1]
        assert in_dim == self.in_dim
        x = hidden_state.reshape(-1, in_dim)
        y = torch.nn.functional.linear(x, self.fc1_weight, self.fc1_bias)
        y = self.act_fn(y)
        y = torch.nn.functional.linear(y, self.fc2_weight, self.fc2_bias)
        y = y.reshape(*hidden_state.shape[:-1], self.out_dim)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def _cuda_forward(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        if out is None:
            out = alloc_func(
                (*hidden_state.shape[:-1], self.out_dim), dtype=hidden_state.dtype, device=hidden_state.device
            )
        result = self._native_forward(hidden_state=hidden_state, out=None, _alloc_func=alloc_func)
        out.copy_(result)
        return out

    def _musa_forward(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._cuda_forward(hidden_state=hidden_state, out=out, alloc_func=alloc_func)

    def __call__(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        return self._forward(hidden_state=hidden_state, out=out, alloc_func=alloc_func)
