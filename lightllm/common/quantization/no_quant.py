import torch
from typing import Optional, List, Union, Tuple

from lightllm.common.quantization.quantize_method import QuantizationMethod, WeightPack
from lightllm.common.quantization.registry import QUANTMETHODS


@QUANTMETHODS.register("none", platform="musa")
@QUANTMETHODS.register("none", platform="cuda")
class NoQuantization(QuantizationMethod):
    """No quantization - uses full precision weights."""

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: WeightPack,
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        weight = weight_pack.weight.t()
        target_dtype = out_dtype if out_dtype is not None else input_tensor.dtype
        if out is None:
            shape = (input_tensor.shape[0], weight.shape[1])
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, target_dtype, device=device)
            else:
                out = torch.empty(shape, dtype=target_dtype, device=device)
        else:
            assert out.dtype == target_dtype, (
                f"NoQuantization.apply: pre-allocated out.dtype={out.dtype} does not match "
                f"requested out_dtype={target_dtype}"
            )
        if bias is None:
            return torch.mm(input_tensor, weight, out=out, out_dtype=target_dtype)
        assert out_dtype is None, "NoQuantization.apply: out_dtype not supported when bias is set"
        return torch.addmm(bias, input_tensor, weight, out=out)

    def _create_weight(
        self, out_dims: Union[int, List[int]], in_dim: int, dtype: torch.dtype, device_id: int, num_experts: int = 1
    ) -> Tuple[WeightPack, List[WeightPack]]:
        out_dim = sum(out_dims) if isinstance(out_dims, list) else out_dims
        expert_prefix = (num_experts,) if num_experts > 1 else ()
        weight = torch.empty(expert_prefix + (out_dim, in_dim), dtype=dtype).cuda(device_id)
        mm_param = WeightPack(weight=weight, weight_scale=None, weight_zero_point=None)
        # weight layout is (out_dim, in_dim), so the split dimension is -2.
        mm_param_list = self._split_weight_pack(
            mm_param,
            weight_out_dims=out_dims,
            weight_split_dim=-2,
        )
        return mm_param, mm_param_list

    def _check_weight_need_quanted(self, weight: torch.Tensor) -> bool:
        return False

    def quantize(self, weight: torch.Tensor, output: WeightPack, offset: int = 0) -> None:
        return

    @property
    def method_name(self):
        return "none"
