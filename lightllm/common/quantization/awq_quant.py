import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops, cutlass_scaled_mm
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL, light_ops
from typing import Any
from typing import TYPE_CHECKING, Optional, Tuple
from lightllm.utils.dist_utils import get_current_device_id

if TYPE_CHECKING:
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightPack

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


class AWQBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm are not installed, you can't use quant api of them."
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager

    def quantize(self, weight: torch.Tensor):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    @property
    def method_name(self):
        return "awq-base"


@QUANTMETHODS.register("awq")
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

    def quantize(self, weight: torch.Tensor):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        qzeros = weight_pack.weight_zero_point
        bias = weight_pack.bias

        NEED_DEQUANT_WEIGHT = input_tensor.shape[:-1].numel() >= 256
        if NEED_DEQUANT_WEIGHT:
            fpweight = awq_dequantize(qweight, weight_scale, qzeros, 0, 0, 0)
            out = torch.matmul(input_tensor, fpweight)
        else:
            out = awq_gemm(input_tensor, qweight, weight_scale, qzeros, self.pack_factor)

        if bias is not None:
            out.add_(bias)
        return out


@QUANTMETHODS.register("awq_marlin")
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

    @property
    def method_name(self):
        return "awq_marlin"

    def quantize(self, weight: torch.Tensor):
        raise NotImplementedError("AWQ online quantization is not supported yet.")

    def params_need_repack(self) -> bool:
        """
        用于说明是否需要对量化后的权重进行repack操作，目前只有awq支持
        """
        return True

    def params_repack(
        self, weight: torch.Tensor, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, dtype_type: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一些量化方法在将参数完成量化后，为了加速性能，还需要将参数进行重拍，使算子性能达到最优，如awq方法。
        """
        weight = self._process_weight_after_loading(weight.cuda(get_current_device_id()))
        weight_scale = self._process_weight_scale_after_loading(
            weight_scale.cuda(get_current_device_id()).to(dtype_type)
        )
        weight_zero_point = self._process_weight_zero_point_after_loading(
            weight_zero_point.cuda(get_current_device_id())
        )
        return weight, weight_scale, weight_zero_point

    def _process_weight_after_loading(self, weight: torch.Tensor) -> torch.Tensor:
        assert self.hf_quantization_config is not None, "hf_quantization_config is not set"
        self.k = weight.shape[0]
        self.n = weight.shape[1] * self.pack_factor
        return vllm_ops.awq_marlin_repack(
            weight,
            size_k=weight.shape[0],
            size_n=weight.shape[1] * self.pack_factor,
            num_bits=self.hf_quantization_config["bits"],
        )

    def _process_weight_scale_after_loading(self, weight_scale: torch.Tensor) -> torch.Tensor:
        assert self.hf_quantization_config is not None, "hf_quantization_config is not set"
        group_size = self.hf_quantization_config["group_size"]
        return marlin_permute_scales(
            weight_scale,
            size_k=weight_scale.shape[0] * group_size,
            size_n=weight_scale.shape[1],
            group_size=self.hf_quantization_config["group_size"],
        )

    def _process_weight_zero_point_after_loading(self, weight_zero_point: torch.Tensor) -> torch.Tensor:
        return awq_to_marlin_zero_points(
            weight_zero_point,
            size_k=weight_zero_point.shape[0],
            size_n=weight_zero_point.shape[1] * self.pack_factor,
            num_bits=self.hf_quantization_config["bits"],
        )

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        qzeros = weight_pack.weight_zero_point
        bias = weight_pack.bias
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
