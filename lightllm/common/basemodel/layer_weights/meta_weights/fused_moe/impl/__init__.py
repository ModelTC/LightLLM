from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM


def create_fuse_moe_impl(
    *,
    n_routed_experts: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    quant_method: QuantizationMethod,
    enable_ep_moe: bool = False,
):
    if enable_ep_moe:
        impl_cls = FuseMoeDeepGEMM
    elif quant_method.method_name == "awq_marlin":
        impl_cls = FuseMoeMarlin
    else:
        impl_cls = FuseMoeTriton
    kwargs = dict(
        n_routed_experts=n_routed_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        quant_method=quant_method,
    )
    return impl_cls(**kwargs)
