from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM
from ..expert_parallel_state import ExpertParallelState


def create_fuse_moe_impl(
    *,
    n_routed_experts: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    quant_method: QuantizationMethod,
    expert_parallel_state: ExpertParallelState | None = None,
):
    if expert_parallel_state is not None:
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
    if expert_parallel_state is not None:
        kwargs["expert_parallel_state"] = expert_parallel_state
    return impl_cls(**kwargs)
