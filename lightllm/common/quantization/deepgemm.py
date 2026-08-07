import torch
from typing import Optional, List, Union, Tuple

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import is_sm100_gpu

logger = init_logger(__name__)

try:
    import deep_gemm

    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False


def pack_ue8m0_scales_to_int32(scales: torch.Tensor) -> torch.Tensor:
    """Pack raw UE8M0 scale bytes in DeepGEMM's native int32 byte order.

    ``deep_gemm.pack_ue8m0_to_int`` converts four E8M0 exponent bytes by
    reinterpreting their contiguous storage as one int32.  Native MXFP4
    checkpoints already store those bytes as float8_e8m0fnu (or, in older
    exports, as an int8/uint8 byte view), so no floating-point conversion is
    needed or allowed here.
    """
    if scales.ndim != 2:
        raise ValueError(f"MXFP4 scales must be 2D [N, K/32], got shape {tuple(scales.shape)}")
    supported_dtypes = (torch.float8_e8m0fnu, torch.uint8, torch.int8)
    if scales.dtype not in supported_dtypes:
        raise TypeError("MXFP4 scales must use torch.float8_e8m0fnu or an int8/uint8 byte view, " f"got {scales.dtype}")
    if scales.shape[-1] % 4 != 0:
        raise ValueError(f"MXFP4 scale K dimension must be divisible by 4 for UE8M0 packing, got {scales.shape[-1]}")
    # This is deliberately a byte reinterpretation: on CUDA and CPU it has
    # the same little-endian layout as DeepGEMM's uint8.view(int32) helper.
    return scales.view(torch.uint8).contiguous().view(scales.shape[0], -1).view(torch.int32)


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
        weight, scale = weight_quant(weight.cuda(device), self.block_size, use_ue8m0_scales=is_sm100_gpu())
        output.weight.copy_(weight)
        output.weight_scale.copy_(scale)
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
                use_ue8m0_scales=is_sm100_gpu(),
            )

        if out is None:
            out = alloc_func((m, n), dtype=input_tensor.dtype, device=input_tensor.device)
        _deepgemm_fp8_nt((qinput_tensor, input_scale), (qweight, weight_scale), out)
        return out

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
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
        scale_layout_k = self.block_size * 4  # Each int32 packs four UE8M0 scales.
        assert (
            in_dim % scale_layout_k == 0
        ), f"FP4 required scale layout needs input dimension divisible by {scale_layout_k}"
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim // 2), dtype=torch.int8).cuda(device_id)
        scale_dim = in_dim // scale_layout_k
        weight_scale = torch.empty_strided(
            expert_prefix + (out_dim, scale_dim),
            (out_dim * scale_dim, 1, out_dim) if num_experts > 1 else (1, out_dim),
            dtype=torch.int32,
            device=f"cuda:{device_id}",
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


@QUANTMETHODS.register("mxfp4fp8-b32-deepgemm", platform="cuda")
class DeepGEMMNativeMXFP4B32QuantizationMethod(DeepGEMMFP8FP4B32QuantizationMethod):
    """Load native MXFP4 checkpoint bytes directly into DeepGEMM FP4 buffers."""

    def __init__(self):
        super().__init__()
        # DeepSeek-V4 uses the checkpoint-native names, unlike the online
        # FP4 quantizer above whose scale is generated during loading.
        self.weight_scale_suffix = "scale"

    @property
    def method_name(self):
        # The execution path remains the existing DeepGEMM Mega MoE backend;
        # only this registry key identifies the checkpoint source format.
        return "fp4fp8-b32-deepgemm"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("mxfp4fp8-b32-deepgemm only loads native packed MXFP4 expert weights")

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight.dtype not in (torch.int8, torch.uint8):
            raise TypeError(
                "native MXFP4 packed weights must use torch.int8 or torch.uint8 E2M1 bytes, " f"got {weight.dtype}"
            )
        if tuple(weight.shape) != tuple(weight_pack.weight.shape):
            raise ValueError(
                "native MXFP4 packed weight shape mismatch: "
                f"expected {tuple(weight_pack.weight.shape)}, got {tuple(weight.shape)}"
            )
        # Reinterpret uint8 source bytes as int8 before copying so E2M1 bits
        # are preserved without relying on a numeric dtype conversion.
        weight_pack.weight.copy_(weight.view(torch.int8))
        weight_pack.load_ok[0] = True

    def load_weight_scale(self, weight_scale: torch.Tensor, weight_pack: WeightPack) -> None:
        if weight_scale is None:
            return
        expected_shape = (*weight_pack.weight_scale.shape[:-1], weight_pack.weight_scale.shape[-1] * 4)
        if tuple(weight_scale.shape) != expected_shape:
            raise ValueError(
                "native MXFP4 scale shape mismatch: "
                f"expected {expected_shape} [N, K/32], got {tuple(weight_scale.shape)}"
            )
        packed_scale = pack_ue8m0_scales_to_int32(weight_scale)
        if packed_scale.shape != weight_pack.weight_scale.shape:
            raise ValueError(
                "packed native MXFP4 scale shape mismatch: "
                f"expected {tuple(weight_pack.weight_scale.shape)}, got {tuple(packed_scale.shape)}"
            )
        weight_pack.weight_scale.copy_(packed_scale)
        weight_pack.load_ok[1] = True


@QUANTMETHODS.register(["mxfp4w4a16-b32-marlin"], platform="cuda")
class MXFP4MoEQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        self.block_size = 32
        self.weight_suffix = "weight"
        self.weight_zero_point_suffix = None
        self.weight_scale_suffix = "scale"
        self.has_weight_scale = True
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "mxfp4w4a16-b32-marlin"

    def quantize(self, weight: torch.Tensor, output: WeightPack):
        raise NotImplementedError("mxfp4w4a16-b32-marlin only loads pre-packed MXFP4 expert weights")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "WeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("mxfp4w4a16-b32-marlin is only implemented for fused MoE expert weights")

    def _probe_marlin_layout(self, size_n: int, size_k: int, dtype: torch.dtype, device_id: int):
        """用零输入走一遍真实的 per-expert repack 路径,探出 marlin 终态布局的形状与类型。
        只调用 finalize 同款的 vllm 函数,不复刻其内部公式,杜绝形状漂移。结果按维度缓存
        (各 MoE 层同维,全程只探两次: w13 一次、w2 一次)。"""
        cache_key = (size_n, size_k, dtype)
        cache = getattr(self, "_marlin_layout_cache", None)
        if cache is None:
            cache = self._marlin_layout_cache = {}
        if cache_key in cache:
            return cache[cache_key]

        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            get_marlin_input_dtype,
            marlin_permute_scales,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            mxfp4_marlin_process_scales,
        )

        input_dtype = get_marlin_input_dtype()
        is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
        device = f"cuda:{device_id}"
        qweight = torch.zeros((size_n, size_k // 2), dtype=torch.int8, device=device).view(torch.int32).T.contiguous()
        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=torch.empty(0, dtype=torch.int, device=device),
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        scale = torch.zeros((size_k // self.block_size, size_n), dtype=dtype, device=device)
        marlin_scale = marlin_permute_scales(
            s=scale, size_k=size_k, size_n=size_n, group_size=self.block_size, is_a_8bit=is_a_8bit
        )
        marlin_scale = mxfp4_marlin_process_scales(marlin_scale, input_dtype=input_dtype)
        layout = (
            (tuple(marlin_qweight.shape), marlin_qweight.dtype),
            (tuple(marlin_scale.shape), marlin_scale.dtype),
        )
        cache[cache_key] = layout
        return layout

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        assert in_dim % self.block_size == 0, "MXFP4 scale dimension must be divisible by block_size"
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        # CPU 暂存区: load_hf_weights 灌入原始预打包 MXFP4,finalize 时 repack 进 CUDA 终态。
        weight = torch.empty(expert_prefix + (out_dim, in_dim // 2), dtype=torch.int8, device="cpu")
        weight_scale = torch.empty(
            expert_prefix + (out_dim, in_dim // self.block_size), dtype=torch.float8_e8m0fnu, device="cpu"
        )
        mm_param = WeightPack(weight=weight, weight_scale=weight_scale)
        # CUDA 终态(marlin 布局)在构造期物化,使 mem manager 的 profile 看到真实权重占用
        # ("构造即分配、load 只灌数"的框架契约,与其它 quant 方法一致;惰性到 finalize 才
        # 进卡会让空卡 profile 把 kv 池撑到挤爆权重加载)。finalize 时 repack 结果拷入。
        (w_shape, w_dtype), (s_shape, s_dtype) = self._probe_marlin_layout(out_dim, in_dim, dtype, device_id)
        mm_param.marlin_weight = torch.empty((num_experts,) + w_shape, dtype=w_dtype, device=f"cuda:{device_id}")
        mm_param.marlin_weight_scale = torch.empty((num_experts,) + s_shape, dtype=s_dtype, device=f"cuda:{device_id}")
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
            weight_scale_out_dims=out_dims,
            weight_scale_split_dim=-2,
        )
        return mm_param, mm_param_list

    def finalize_moe_weight(self, moe_weight):
        try:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
                prepare_moe_mxfp4_layer_for_marlin,
            )
        except Exception as e:
            raise RuntimeError(f"mxfp4w4a16-b32-marlin requires vLLM MXFP4 packing utilities, error={repr(e)}") from e

        class _MXFP4Layer:
            pass

        device = torch.device("cuda", moe_weight.device_id_)
        layer = _MXFP4Layer()
        layer.params_dtype = moe_weight.data_type_
        w13 = moe_weight.w13.weight.view(torch.uint8).to(device=device, non_blocking=True).contiguous()
        w2 = moe_weight.w2.weight.view(torch.uint8).to(device=device, non_blocking=True).contiguous()
        w13_scale = moe_weight.w13.weight_scale.to(device=device, non_blocking=True).contiguous()
        w2_scale = moe_weight.w2.weight_scale.to(device=device, non_blocking=True).contiguous()
        (
            w13_new,
            w2_new,
            w13_scale_new,
            w2_scale_new,
            _,
            _,
        ) = prepare_moe_mxfp4_layer_for_marlin(layer, w13, w2, w13_scale, w2_scale, None, None)
        # repack 结果拷入构造期预分配的 marlin 终态 buffer(与 AWQ marlin 路径同形态),
        # CPU 暂存与 repack 临时随引用释放;shape 失配会在 copy_ 处显式报错(探针保证一致)。
        moe_weight.w13.marlin_weight.copy_(w13_new)
        moe_weight.w13.marlin_weight_scale.copy_(w13_scale_new)
        moe_weight.w2.marlin_weight.copy_(w2_new)
        moe_weight.w2.marlin_weight_scale.copy_(w2_scale_new)
        moe_weight.w13.weight = moe_weight.w13.marlin_weight
        moe_weight.w13.weight_scale = moe_weight.w13.marlin_weight_scale
        moe_weight.w2.weight = moe_weight.w2.marlin_weight
        moe_weight.w2.weight_scale = moe_weight.w2.marlin_weight_scale


def _deepgemm_fp8_nt(a_tuple, b_tuple, out):
    if HAS_DEEPGEMM:
        if hasattr(deep_gemm, "gemm_fp8_fp8_bf16_nt"):
            return deep_gemm.gemm_fp8_fp8_bf16_nt([a_tuple[0], a_tuple[1]], [b_tuple[0], b_tuple[1]], out)
        if hasattr(deep_gemm, "fp8_gemm_nt"):
            return deep_gemm.fp8_gemm_nt((a_tuple[0], a_tuple[1]), (b_tuple[0], b_tuple[1]), out)
    raise RuntimeError("deep_gemm does not provide fp8 NT GEMM kernel in this version")
