import torch

from lightllm.common.basemodel.layer_weights.meta_weights import GatedRMSNormWeight
from lightllm.models.qwen4_exp.triton_kernel.gated_norm import gated_norm


class Qwen4ExpGatedNormWeight(GatedRMSNormWeight):
    def __init__(self, dim, weight_name, data_type, activation="sigmoid"):
        if activation not in ("sigmoid", "silu"):
            raise ValueError("Qwen4Exp GDN output gate must be sigmoid or silu")
        self.activation = activation
        super().__init__(dim, weight_name, data_type)

    def _triton_forward(self, input, gate_value, eps, out=None, alloc_func=torch.empty):
        if out is None:
            out = alloc_func(input.shape, dtype=input.dtype, device=input.device)
        return gated_norm(input, gate_value, self.weight, eps, self.activation, out)

    def _native_forward(self, input, gate_value, eps, out=None, alloc_func=torch.empty):
        values = input.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        weighted = self.weight * normalized.to(input.dtype)
        gate = gate_value.float().sigmoid()
        if self.activation == "silu":
            gate = gate * gate_value.float()
        result = (weighted * gate).to(input.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result
