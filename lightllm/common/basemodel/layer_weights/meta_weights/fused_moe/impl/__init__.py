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
    """创建持有自身路由运行态的 MoE 执行实现。

    这里直接返回完成初始化的对象，而不是仅返回实现类，使 EPLB 布局、路由
    计数器等后端专属状态与使用它们的 kernel 保持在同一个实现对象中。
    """
    if enable_ep_moe:
        impl_cls = FuseMoeDeepGEMM
    elif quant_method.method_name == "awq_marlin":
        impl_cls = FuseMoeMarlin
    else:
        impl_cls = FuseMoeTriton

    return impl_cls(
        n_routed_experts=n_routed_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        quant_method=quant_method,
    )
