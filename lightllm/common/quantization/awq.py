import torch
from typing import Any, Optional, Tuple, List

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

    if HAS_VLLM:
        awq_dequantize = vllm_ops.awq_dequantize
        awq_gemm = vllm_ops.awq_gemm
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            marlin_permute_scales,
            awq_to_marlin_zero_points,
            should_use_atomic_add_reduce,
            marlin_make_empty_g_idx,
            marlin_make_workspace_new,
        )
        from vllm.scalar_type import scalar_types

        TYPE_MAP = {
            4: scalar_types.uint4,
            8: scalar_types.uint8,
        }
    else:
        awq_dequantize = None
        awq_gemm = None
        TYPE_MAP = {}
except ImportError:
    HAS_VLLM = False
    awq_dequantize = None
    awq_gemm = None
    TYPE_MAP = {}


class AWQBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm are not installed, you can't use quant api of them."
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    @property
    def method_name(self):
        return "awq-base"


@QUANTMETHODS.register("awq", platform="cuda")
class AWQW4A16QuantizationMethod(AWQBaseQuantizationMethod):
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

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        qzeros = weight_pack.weight_zero_point

        NEED_DEQUANT_WEIGHT = input_tensor.shape[:-1].numel() >= 256
        if NEED_DEQUANT_WEIGHT:
            fpweight = awq_dequantize(qweight, weight_scale, qzeros, 0, 0, 0)
            out = torch.matmul(input_tensor, fpweight)
        else:
            out = awq_gemm(input_tensor, qweight, weight_scale, qzeros, self.pack_factor)

        if bias is not None:
            out.add_(bias)
        return out

    def _create_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims)
        group_size = self.hf_quantization_config["group_size"]
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (in_dim, out_dim // self.pack_factor), dtype=torch.int32).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (in_dim // group_size, out_dim), dtype=dtype).cuda(device_id)
        weight_zero_point = torch.empty(
            expert_prefix + (in_dim // group_size, out_dim // self.pack_factor), dtype=torch.int32
        ).cuda(device_id)
        weight_out_dims = [_out_dim // self.pack_factor for _out_dim in out_dims]
        weight_scale_out_dims = out_dims
        weight_zero_point_out_dims = weight_out_dims
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=weight_out_dims,
            weight_split_dim=-1,
            weight_scale_out_dims=weight_scale_out_dims,
            weight_scale_split_dim=-1,
            weight_zero_point_out_dims=weight_zero_point_out_dims,
            weight_zero_point_split_dim=-1,
        )
        return mm_param, mm_param_list


@QUANTMETHODS.register("awq_marlin", platform="cuda")
class AWQMARLINW4A16QuantizationMethod(AWQBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.pack_factor = 8
        self.nbits = 4
        self.weight_scale_suffix = "scales"
        self.weight_zero_point_suffix = "qzeros"
        self.weight_suffix = "qweight"
        self.g_idx = marlin_make_empty_g_idx(torch.device("cuda"))
        self.g_idx_sort_indices = marlin_make_empty_g_idx(torch.device("cuda"))
        self.workspace = marlin_make_workspace_new(torch.device("cuda"))
        self.vllm_quant_type = TYPE_MAP[self.nbits]
        self.has_weight_scale = True
        self.has_weight_zero_point = True
        self.tile_size = 16

    @property
    def method_name(self):
        return "awq_marlin"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        qzeros = weight_pack.weight_zero_point
        reshaped_x = input_tensor.reshape(-1, input_tensor.shape[-1])

        use_atomic_add = should_use_atomic_add_reduce(
            m=reshaped_x.size(0),
            n=self.n,
            k=self.k,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        out = vllm_ops.gptq_marlin_gemm(
            reshaped_x,
            None,
            qweight,
            bias,
            weight_scale,
            None,
            qzeros,
            self.g_idx,
            self.g_idx_sort_indices,
            self.workspace,
            self.vllm_quant_type,
            size_m=reshaped_x.shape[0],
            size_n=self.n,
            size_k=self.k,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        if bias is not None:
            out.add_(bias)
        return out

    def _create_weight(
        self, out_dims: List[int], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims)
        self.n = out_dim
        self.k = in_dim
        group_size = self.hf_quantization_config["group_size"]
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(
            expert_prefix + (in_dim // self.tile_size, out_dim * self.tile_size // self.pack_factor), dtype=torch.int32
        ).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (in_dim // group_size, out_dim), dtype=dtype).cuda(device_id)
        weight_zero_point = torch.empty(
            expert_prefix + (in_dim // group_size, out_dim // self.pack_factor), dtype=torch.int32
        ).cuda(device_id)
        weight_out_dims = [_out_dim * self.tile_size // self.pack_factor for _out_dim in out_dims]
        weight_scale_out_dims = out_dims
        weight_zero_point_out_dims = [_out_dim // self.pack_factor for _out_dim in out_dims]
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale, weight_zero_point=weight_zero_point)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=weight_out_dims,
            weight_split_dim=-1,
            weight_scale_out_dims=weight_scale_out_dims,
            weight_scale_split_dim=-1,
            weight_zero_point_out_dims=weight_zero_point_out_dims,
            weight_zero_point_split_dim=-1,
        )
        return mm_param, mm_param_list

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        assert self.hf_quantization_config is not None, "hf_quantization_config is not set"
        if weight is None:
            return
        device_id = get_current_device_id()
        repack_weight = vllm_ops.awq_marlin_repack(
            weight.cuda(device_id),
            size_k=weight.shape[0],
            size_n=weight.shape[1] * self.pack_factor,
            num_bits=self.hf_quantization_config["bits"],
        )
        weight_pack.weight.copy_(repack_weight)
        weight_pack.load_ok[0] = True
        return

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack) -> None:
        assert self.hf_quantization_config is not None, "hf_quantization_config is not set"
        if weight_scale is None:
            return
        group_size = self.hf_quantization_config["group_size"]
        device_id = get_current_device_id()
        repack_weight_scale = marlin_permute_scales(
            weight_scale.cuda(device_id),
            size_k=weight_scale.shape[0] * group_size,
            size_n=weight_scale.shape[1],
            group_size=self.hf_quantization_config["group_size"],
        )
        weight_pack.weight_scale.copy_(repack_weight_scale)
        weight_pack.load_ok[1] = True
        return

    def load_weight_zero_point(self, weight_zero_point: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight_zero_point is None:
            return
        device_id = get_current_device_id()
        repack_weight_zero_point = awq_to_marlin_zero_points(
            weight_zero_point.cuda(device_id),
            size_k=weight_zero_point.shape[0],
            size_n=weight_zero_point.shape[1] * self.pack_factor,
            num_bits=self.hf_quantization_config["bits"],
        )
        weight_pack.weight_zero_point.copy_(repack_weight_zero_point)
        weight_pack.load_ok[2] = True
        return


# adapted from
# https://github.com/vllm-project/vllm/blob/aef368aa08572505b820db01da82e2fbb3d43a72/vllm/model_executor/layers/quantization/awq_marlin.py#L211-L212
def is_awq_marlin_compatible(quantization_config: dict[str, Any]):
    # Extract data from quant config.
    quant_method = quantization_config.get("quant_method", "").lower()
    num_bits = quantization_config.get("bits")
    group_size = quantization_config.get("group_size")
    zero_point = quantization_config.get("zero_point")

    if not torch.cuda.is_available():
        return False

    if quant_method != "awq":
        return False

    # If we cannot find the info needed in the config, cannot convert.
    if num_bits is None or group_size is None or zero_point is None:
        return False

    if num_bits not in TYPE_MAP:
        return False

    return check_marlin_supported(quant_type=TYPE_MAP[num_bits], group_size=group_size, has_zp=zero_point)
