import math
import torch
import numpy as np
from typing import Dict, Optional
from transformers.activations import ACT2FN
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.basemodel.layer_weights.meta_weights.platform_op import PlatformAwareOp
from lightllm.common.basemodel.triton_kernel.embedding import embedding as embedding_kernel
from lightllm.utils.dist_utils import get_dp_world_size, get_current_rank_in_dp


class Qwen3OmniMoeCode2wavCausalTransConvNetWeight(BaseWeightTpl, PlatformAwareOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        weight_name: str,
        bias_name: str,
        data_type: torch.dtype,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_name: str = weight_name
        self.bias_name: str = bias_name
        self.data_type_ = data_type

        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad = self.left_pad

        self._create_weight()

    def _create_weight(self):
        # ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size) when groups=1
        self.weight: torch.Tensor = torch.empty(
            self.in_channels, self.out_channels, self.kernel_size, dtype=self.data_type_, device=self.device_id_
        )
        self.bias: torch.Tensor = torch.empty(self.out_channels, dtype=self.data_type_, device=self.device_id_)
        self.weight.load_ok = False
        self.bias.load_ok = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]):
        if self.weight_name in weights:
            t_weight = weights[self.weight_name]
            assert t_weight.shape == (self.in_channels, self.out_channels, self.kernel_size)
            self.weight.copy_(t_weight.to(self.data_type_))
            self.weight.load_ok = True
        if self.bias_name in weights:
            t_bias = weights[self.bias_name]
            assert t_bias.shape == (self.out_channels,)
            self.bias.copy_(t_bias.to(self.data_type_))
            self.bias.load_ok = True

    def verify_load(self):
        return self.weight.load_ok and self.bias.load_ok

    def _native_forward(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, _alloc_func=torch.empty
    ) -> torch.Tensor:
        # hidden_state: [B, C_in, L]
        x = torch.nn.functional.conv_transpose1d(
            hidden_state,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
        )
        x = x[..., self.left_pad : x.shape[-1] - self.right_pad].contiguous()
        if out is not None:
            out.copy_(x)
            return out
        return x

    def _cuda_forward(
        self, hidden_state: torch.Tensor, out: Optional[torch.Tensor] = None, alloc_func=torch.empty
    ) -> torch.Tensor:
        if out is None:
            # output length depends on input length; allocate after computing
            result = self._native_forward(hidden_state=hidden_state, out=None, _alloc_func=alloc_func)
            return result
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
