import os
import torch
from typing import Optional, List, Union, Tuple

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    import deep_gemm

    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False


class DeepGEMMBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        self.cache_manager = g_cache_manager
        assert HAS_DEEPGEMM, "deepgemm is not installed, you can't use quant api of it"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("Not implemented")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Not implemented")

    @property
    def method_name(self):
        return "base-deepgemm"


@QUANTMETHODS.register("fp8w8a8-b128-deepgemm", platform="cuda")
class DeepGEMMFP8w8a8B128QuantizationMethod(DeepGEMMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 128
        self.weight_suffix = "weight"
        self.weight_zero_point_suffix = None
        self.weight_scale_suffix = "weight_scale_inv"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "fp8w8a8-b128-deepgemm"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import weight_quant

        device = output.weight.device
        weight, scale = weight_quant(weight.cuda(device), self.block_size, use_ue8m0_scales=self.use_ue8m0_scales)
        output.weight.copy_(weight)
        self.load_weight_scale(scale, output)
        return

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight_scale is None:
            return
        repack_weight_scale = weight_scale
        if self.use_packed_ue8m0 and weight_scale.dtype == torch.float32:
            n, k = weight_pack.weight.shape[-2:]
            repack_weight_scale = deep_gemm.transform_sf_into_required_layout(
                weight_scale.cuda(weight_pack.weight.device),
                n,
                k,
                (1, self.block_size, self.block_size),
                num_groups=weight_scale.shape[0] if weight_scale.ndim == 3 else None,
                is_sfa=False,
            )
        weight_pack.weight_scale.copy_(repack_weight_scale)
        weight_pack.load_ok[1] = True

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "WeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = weight_pack.weight
        weight_scale = weight_pack.weight_scale
        input_scale = None
        alloc_func = torch.empty if not use_custom_tensor_mananger else self.cache_manager.empty
        m, k = input_tensor.shape
        n = qweight.shape[0]
        if input_scale is None:
            qinput_tensor, input_scale = per_token_group_quant_fp8(
                input_tensor,
                self.block_size,
                dtype=qweight.dtype,
                column_major_scales=True,
                scale_tma_aligned=True,
                alloc_func=alloc_func,
                use_ue8m0_scales=self.use_ue8m0_scales,
                use_packed_ue8m0=self.use_packed_ue8m0,
            )

        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        _deepgemm_fp8_nt((qinput_tensor, input_scale), (qweight, weight_scale), out)
        return out

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        self.use_ue8m0_scales = (
            self.hf_quantization_config is not None and self.hf_quantization_config.get("scale_fmt") == "ue8m0"
        ) or os.getenv("LIGHTLLM_USE_UE8M0_SCALES", "0").upper() in ["ON", "TRUE", "1"]
        self.use_packed_ue8m0 = self.use_ue8m0_scales and is_sm100_gpu()
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        weight_scale_out_dims = [(_out_dim + self.block_size - 1) // self.block_size for _out_dim in out_dims]
        divisible_by_block_size = [_out_dim % self.block_size != 0 for _out_dim in out_dims]
        if sum(divisible_by_block_size) > 1:
            raise ValueError(
                f"out_dims only contains one dim can not be divisible \
                by block_size {self.block_size}, but got {out_dims}"
            )
        weight_scale_out_dim = sum(weight_scale_out_dims)
        weight_scale_in_dim = (in_dim + self.block_size - 1) // self.block_size
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=torch.float8_e4m3fn).cuda(device_id)
        if self.use_packed_ue8m0:
            aligned_out_dim = (out_dim + 3) // 4 * 4
            weight_scale = torch.empty(
                expert_prefix + ((weight_scale_in_dim + 3) // 4, aligned_out_dim),
                dtype=torch.int32,
                device=f"cuda:{device_id}",
            ).transpose(-1, -2)[..., :out_dim, :]
            weight_scale_out_dims = out_dims
        else:
            weight_scale = torch.empty(
                expert_prefix + (weight_scale_out_dim, weight_scale_in_dim), dtype=torch.float32
            ).cuda(device_id)
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=weight_scale_out_dims,
            weight_scale_split_dim=-2,
        )
        return mm_param, mm_param_list


@QUANTMETHODS.register("fp4fp8-b32-deepgemm", platform="cuda")
class DeepGEMMFP8FP4B32QuantizationMethod(DeepGEMMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 32
        self.weight_suffix = "weight"
        self.weight_zero_point_suffix = None
        self.weight_scale_suffix = None
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "fp4fp8-b32-deepgemm"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        from deep_gemm.utils import per_token_cast_to_fp4
        import deep_gemm

        weight = weight.cuda(output.weight.device)
        if weight.dim() == 2:
            n, k = weight.shape
            packed_weight, weight_scale = per_token_cast_to_fp4(weight, use_ue8m0=True, gran_k=self.block_size)
            weight_scale = deep_gemm.transform_sf_into_required_layout(weight_scale, n, k, (1, self.block_size), None)
        else:
            num_groups, n, k = weight.shape
            packed_weight = torch.empty((num_groups, n, k // 2), device=weight.device, dtype=torch.int8)
            weight_scale = torch.empty((num_groups, n, k // self.block_size), device=weight.device, dtype=torch.float32)
            for i in range(num_groups):
                packed_weight[i], weight_scale[i] = per_token_cast_to_fp4(
                    weight[i], use_ue8m0=True, gran_k=self.block_size
                )
            weight_scale = deep_gemm.transform_sf_into_required_layout(
                weight_scale, n, k, (1, self.block_size), num_groups
            )
        output.weight.copy_(packed_weight)
        output.weight_scale.copy_(weight_scale)
        return

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "WeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("fp4fp8-b32-deepgemm is only implemented for fused MoE expert weights")

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        assert in_dim % 2 == 0, "FP4 packed weight requires even input dimension"
        assert in_dim % self.block_size == 0, "FP4 scale dimension must be divisible by block_size"
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim // 2), dtype=torch.int8).cuda(device_id)
        weight_scale = torch.empty(expert_prefix + (out_dim, in_dim // self.block_size), dtype=torch.int32).cuda(
            device_id
        )
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-2,
        )
        return mm_param, mm_param_list


def _deepgemm_fp8_nt(a_tuple, b_tuple, out):
    if HAS_DEEPGEMM:
        if hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
            return deep_gemm.gemm_fp8_fp8_bf16_nt([a_tuple[0], a_tuple[1]], [b_tuple[0], b_tuple[1]], out)
        if hasattr(deep_gemm, "fp8_gemm_nt"):
            return deep_gemm.fp8_gemm_nt((a_tuple[0], a_tuple[1]), (b_tuple[0], b_tuple[1]), out)
    raise RuntimeError("deep_gemm does not provide fp8 NT GEMM kernel in this version")


@QUANTMETHODS.register("mxfp4w4a16-b32-marlin", platform="cuda")
class MXFP4MoEQuantizationMethod(QuantizationMethod):
    """Checkpoint MXFP4 experts, repacked once for Hopper's weight-only Marlin."""

    def __init__(self):
        super().__init__()
        self.block_size = 32
        self.weight_scale_suffix = "scale"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "mxfp4w4a16-b32-marlin"

    def quantize(self, weight, output):
        raise NotImplementedError("MXFP4 Marlin loads prequantized expert checkpoints")

    def apply(self, *args, **kwargs):
        raise NotImplementedError("MXFP4 Marlin is an expert weight format")

    def _create_weight(self, out_dims, in_dim, dtype, device_id, num_experts=1):
        # TP8 on V4.1 leaves 288 intermediate channels; Marlin needs 128-aligned tiles.
        padded_dims = [(n + 127) // 128 * 128 for n in out_dims]
        padded_k = (in_dim + 127) // 128 * 128
        padded_n = sum(padded_dims)
        weight = torch.zeros((num_experts, padded_n, padded_k // 2), dtype=torch.int8)
        scales = torch.ones((num_experts, padded_n, padded_k // 32), dtype=torch.float32).to(torch.float8_e8m0fnu)
        pack = WeightPack(weight=weight, weight_scale=scales)
        # Reserve runtime bytes before KV-cache profiling; repacking determines the final views.
        device = f"cuda:{device_id}"
        pack.marlin_weight = torch.empty_like(weight.view(torch.int32), device=device)
        pack.marlin_weight_scale = torch.empty_like(scales, device=device)
        children = self._split_weight_pack(
            pack,
            weight_out_dims=padded_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=padded_dims,
            weight_scale_split_dim=-2,
        )
        return pack, children

    def load_weight(self, weight, pack):
        n, k = weight.shape[-2:]
        pack.weight[..., :n, :k].copy_(weight)
        pack.load_ok[0] = True

    def load_weight_scale(self, scale, pack):
        n, k = scale.shape[-2:]
        pack.weight_scale[..., :n, :k].copy_(scale)
        pack.load_ok[1] = True

    def finalize_moe_weight(self, moe_weight):
        from types import SimpleNamespace
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

        device = moe_weight.w13.marlin_weight.device
        w13, w2 = moe_weight.w13, moe_weight.w2
        packed = prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=moe_weight.data_type_),
            w13.weight.view(torch.uint8).to(device),
            w2.weight.view(torch.uint8).to(device),
            w13.weight_scale.to(device),
            w2.weight_scale.to(device),
            None,
            None,
        )
        for pack, weight, scale in ((w13, packed[0], packed[2]), (w2, packed[1], packed[3])):
            pack.weight = pack.marlin_weight.view_as(weight)
            pack.weight_scale = pack.marlin_weight_scale.view_as(scale)
            pack.weight.copy_(weight)
            pack.weight_scale.copy_(scale)
            pack.marlin_weight = pack.marlin_weight_scale = None
        # Drop the CPU staging views held by the loader after the final repack.
        moe_weight.w1_list = moe_weight.w2_list = moe_weight.w3_list = []
        moe_weight.w1 = moe_weight.w3 = None
