import torch
from gguf import GGMLQuantizationType, quant_shape_from_byte_shape
from gguf.quants import quant_shape_to_byte_shape
from typing import List, Optional, Tuple

from .registry import QUANTMETHODS
from .quantize_method import GGUFWeightMeta, QuantizationMethod, WeightPack
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.basemodel.layer_weights.gguf_load_utils import UNQUANTIZED_GGML_TYPES
from lightllm.common.gguf_kernel.dequantization import get_gguf_dequant_fn
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# To avoid logging the warning multiple times
_predquant_warned = False


def _warn_predquant_once(unsupported_quant_type: str) -> None:
    global _predquant_warned
    if _predquant_warned:
        return
    _predquant_warned = True
    logger.warning(
        "The current GGUF model contains quantization formats that do not support runtime "
        f"dequantization (e.g., {unsupported_quant_type}). These weights will be dequantized during model loading, which "
        "may increase GPU memory usage. To add support, register a dequantization implementation via "
        "register_gguf_dequant in lightllm/common/gguf_kernel/dequantization.py.",
    )


def _linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bias is None:
        return torch.mm(input_tensor, weight, out=out)
    return torch.addmm(bias, input_tensor, weight, out=out)


@QUANTMETHODS.register("gguf", platform="cuda")
class GGUFQuantizationMethod(QuantizationMethod):

    def __init__(self):
        super().__init__()

    def quantize(self, weight: torch.Tensor, output: WeightPack) -> None:
        raise NotImplementedError("GGUF online quantization is not supported")

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Allocate output tensor based on the shape of the weight pack
        assert weight_pack.gguf_quant_type is not None, "gguf_quant_type must be set on WeightPack"
        if out is None:
            if (
                weight_pack.gguf_quant_type in UNQUANTIZED_GGML_TYPES
                or weight_pack.gguf_load_predquant
            ):
                out_features = weight_pack.weight.shape[-2]
            else:
                out_features, _ = quant_shape_from_byte_shape(
                    weight_pack.weight.shape,
                    weight_pack.gguf_quant_type,
                )
            shape = (input_tensor.shape[0], out_features)
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, input_tensor.dtype, device=input_tensor.device)
            else:
                out = torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
        # Unquantized types and load-time dequantized weights are directly used
        if (
            weight_pack.gguf_quant_type in UNQUANTIZED_GGML_TYPES
            or weight_pack.gguf_load_predquant
        ):
            weight = weight_pack.weight.t()
            return _linear(input_tensor, weight, out, bias)
        # For quantized types, we need to dequantize the weight and then use it
        dequant_fn = get_gguf_dequant_fn(weight_pack.gguf_quant_type)
        if dequant_fn is None:
            raise ValueError(
                f"Unsupported GGUF quantization type: {weight_pack.gguf_quant_type}"
            )
        m, n = quant_shape_from_byte_shape(weight_pack.weight.shape,
                                           weight_pack.gguf_quant_type)
        alloc_func = torch.empty if not use_custom_tensor_mananger else g_cache_manager.empty
        dequantized_weight = alloc_func((m, n),
                                        dtype=input_tensor.dtype,
                                        device=input_tensor.device)
        dequant_fn(
            weight_pack.weight,
            m,
            n,
            input_tensor.dtype,
            out=dequantized_weight,
        )
        weight = dequantized_weight.t()

        return _linear(input_tensor, weight, out, bias)

    @property
    def method_name(self):
        return "gguf"

    def load_weight(self, weight: torch.Tensor, weight_pack: WeightPack) -> None:
        assert not weight_pack.gguf_load_predquant, (
            "predquant GGUF weights must be loaded via load_weight_shard"
        )
        if weight.shape != weight_pack.weight.shape:
            raise ValueError(
                f"GGUF weight shape {weight.shape} does not match weight pack shape {weight_pack.weight.shape}"
            )
        device = weight_pack.weight.device
        weight_pack.weight.copy_(
            weight.contiguous().to(device=device, dtype=weight_pack.weight.dtype)
        )
        weight_pack.load_ok[0] = True

    def _check_weight_need_quanted(self, weight: torch.Tensor) -> bool:
        return False

    def _create_weight(
        self,
        out_dims: List[int],
        in_dim: int,
        dtype: torch.dtype,
        device_id: int,
        num_experts: int = 1,
        weight_names: Optional[List[str]] = None,
    ) -> Tuple[WeightPack, List[WeightPack]]:
        assert weight_names is not None and len(weight_names) > 0, "weight_names must be provided"
        assert len(weight_names) == len(out_dims), "weight_names and out_dims must align"

        weight_dtype = None
        shard_quant_types: List[GGMLQuantizationType] = []
        for weight_name in weight_names:
            meta: GGUFWeightMeta = self.gguf_quant_meta_map[weight_name]
            shard_quant_types.append(meta.quant_type)
            quant_shape = meta.shape
            assert len(
                quant_shape
            ) == 2, f"GGUF linear weight must be 2D, got {quant_shape} for {weight_name}"
            if weight_dtype is None:
                weight_dtype = meta.dtype
            else:
                assert weight_dtype == meta.dtype, f"merged GGUF weights must share dtype, got {weight_dtype} vs {meta.dtype}"

        # If there are mixed quant types, we need to dequant each shard at load time
        mixed_quant_types = len(set(shard_quant_types)) > 1
        gguf_quant_type = shard_quant_types[0]
        gguf_load_predquant = mixed_quant_types
        if not mixed_quant_types and gguf_quant_type not in UNQUANTIZED_GGML_TYPES:
            if get_gguf_dequant_fn(gguf_quant_type) is None:
                gguf_load_predquant = True
                _warn_predquant_once(gguf_quant_type.name)

        logical_shape_rowmajor = (sum(out_dims), in_dim)
        expert_prefix = (num_experts, ) if num_experts > 1 else ()
        if gguf_quant_type in UNQUANTIZED_GGML_TYPES or gguf_load_predquant:
            full_shape = expert_prefix + logical_shape_rowmajor
            storage_dtype = dtype if gguf_load_predquant else weight_dtype
        else:
            full_shape = expert_prefix + quant_shape_to_byte_shape(
                logical_shape_rowmajor, gguf_quant_type)
            storage_dtype = weight_dtype

        weight = torch.empty(full_shape, dtype=storage_dtype).cuda(device_id)

        mm_param = WeightPack(
            weight=weight,
            weight_scale=None,
            weight_zero_point=None,
            gguf_quant_type=gguf_quant_type,
            gguf_load_predquant=gguf_load_predquant,
        )
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
        )
        for pack, shard_quant_type in zip(mm_param_list, shard_quant_types):
            pack.gguf_quant_type = shard_quant_type

        return mm_param, mm_param_list
