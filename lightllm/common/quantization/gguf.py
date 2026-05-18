import torch
from gguf import GGMLQuantizationType, quant_shape_from_byte_shape
from gguf.quants import quant_shape_to_byte_shape
from typing import List, Optional, Tuple

from .registry import QUANTMETHODS
from .quantize_method import GGUFWeightMeta, QuantizationMethod, WeightPack
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.models.gguf.triton_kernel.dequantization import get_gguf_dequant_fn

# These types are not quantized, so they are directly used
UNQUANTIZED_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16}


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
        # allocate output tensor if not provided
        assert weight_pack.gguf_quant_type is not None, "gguf_quant_type must be set on WeightPack"
        if out is None:
            if weight_pack.gguf_quant_type in UNQUANTIZED_TYPES:
                out_features = weight_pack.weight.shape[0]
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
        # unquantized types are directly used
        if weight_pack.gguf_quant_type in UNQUANTIZED_TYPES:
            weight = weight_pack.weight.t()
            return _linear(input_tensor, weight, out, bias)
        # quantized types are dequantized and then used
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

    def load_weight(self, weight: torch.Tensor,
                    weight_pack: WeightPack) -> None:
        device = weight_pack.weight.device
        weight_pack.weight.copy_(weight.contiguous().to(
            device=device, dtype=weight_pack.weight.dtype))
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
        assert weight_names is not None and len(
            weight_names) > 0, "weight_names must be provided"
        if self.gguf_quant_meta_map is None:
            raise ValueError(
                f"Cannot load GGUF-quantized weights {weight_names!r}: no GGUF metadata was built. "
                f"quant_type is 'gguf', but model_dir has no .gguf file (only HuggingFace/safetensors). "
                f"Use --quant_type to no set gguf for HF checkpoints, or set model_dir to a path that contains "
                f"exactly one .gguf file."
            )

        assert len(weight_names) == len(
            out_dims), "weight_names and out_dims must align"
        weight_dtype = None
        gguf_quant_types = set()
        for weight_name in weight_names:
            meta: GGUFWeightMeta = self.gguf_quant_meta_map[weight_name]
            gguf_quant_types.add(meta.quant_type)
            quant_shape = meta.shape
            assert len(
                quant_shape
            ) == 2, f"GGUF linear weight must be 2D, got {quant_shape} for {weight_name}"
            _, in_d = quant_shape[0], quant_shape[1]
            assert in_d == in_dim, (
                f"GGUF tensor {weight_name} has in_features {in_d}, layer expects in_dim {in_dim}"
            )
            if weight_dtype is None:
                weight_dtype = meta.dtype
            else:
                assert weight_dtype == meta.dtype, f"merged GGUF weights must share dtype, got {weight_dtype} vs {meta.dtype}"
        assert len(
            gguf_quant_types
        ) == 1, f"merged GGUF weights must share quant_type, got {gguf_quant_types}"
        gguf_quant_type = gguf_quant_types.pop()

        if gguf_quant_type not in UNQUANTIZED_TYPES:
            if get_gguf_dequant_fn(gguf_quant_type) is None:
                raise ValueError(
                    f"No CUDA dequant registered for GGUF type {gguf_quant_type!r}; "
                    f"add @register_gguf_dequant in "
                    f"lightllm/models/gguf/triton_kernel/dequantization.py"
                )

        # Buffer sizes follow layer/tp shard dims from the caller (load_path slices file weights into this storage).
        logical_shape_rowmajor = (sum(out_dims), in_dim)
        expert_prefix = (num_experts, ) if num_experts > 1 else ()
        if gguf_quant_type in UNQUANTIZED_TYPES:
            full_shape = expert_prefix + logical_shape_rowmajor
        else:
            full_shape = expert_prefix + quant_shape_to_byte_shape(
                logical_shape_rowmajor, gguf_quant_type)
        weight = torch.empty(full_shape, dtype=weight_dtype).cuda(device_id)
        mm_param = WeightPack(
            weight=weight,
            weight_scale=None,
            weight_zero_point=None,
            gguf_quant_type=gguf_quant_type,
        )
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
        )

        return mm_param, mm_param_list
