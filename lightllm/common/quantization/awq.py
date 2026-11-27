import torch
from typing import Optional, Tuple, List

from lightllm.common.basemodel.triton_kernel.quantization.awq_gemm import awq_gemm
from lightllm.common.quantization.awq_marlin import can_prepare_awq, dense_awq_marlin, is_marlin_packed, prepare_awq
from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS


@QUANTMETHODS.register(["awq", "awq_marlin"], platform="cuda")
class AWQW4A16QuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        self.pack_factor = 8
        self.weight_scale_suffix = "scales"
        self.weight_zero_point_suffix = "qzeros"
        self.weight_suffix = "qweight"
        self.has_weight_scale = True
        self.has_weight_zero_point = True

    @property
    def method_name(self):
        return "awq"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def prepare_dense_weight(self, pack: WeightPack):
        if not is_marlin_packed(pack) and can_prepare_awq(pack, "dense"):
            prepare_awq(pack)

    def prepare_moe_weights(self, gate: WeightPack, down: WeightPack):
        if not is_marlin_packed(gate) and can_prepare_awq(gate, "moe") and can_prepare_awq(down, "moe"):
            prepare_awq(gate)
            prepare_awq(down)

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_marlin_packed(weight_pack):
            result = dense_awq_marlin(input_tensor, weight_pack, out)
        else:
            result = awq_gemm(
                input_tensor, weight_pack.weight, weight_pack.weight_scale, weight_pack.weight_zero_point, out
            )
        if bias is not None:
            result.add_(bias)
        return result

    def _create_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims)
        group_size = self.hf_quantization_config["group_size"]
        if group_size == -1:
            group_size = in_dim
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (in_dim, out_dim // self.pack_factor), dtype=torch.int32).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (in_dim // group_size, out_dim), dtype=dtype).cuda(device_id)
        weight_zero_point = torch.empty(
            expert_prefix + (in_dim // group_size, out_dim // self.pack_factor), dtype=torch.int32
        ).cuda(device_id)
        weight_out_dims = [width // self.pack_factor for width in out_dims]
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=weight_out_dims,
            weight_split_dim=-1,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-1,
            weight_zero_point_out_dims=weight_out_dims,
            weight_zero_point_split_dim=-1,
        )
        return mm_param, mm_param_list
