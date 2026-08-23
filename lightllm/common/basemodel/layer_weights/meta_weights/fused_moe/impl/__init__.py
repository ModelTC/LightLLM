from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM
from .mxfp4_impl import FuseMoeMXFP4
from ..expert_parallel_state import ExpertParallelState


def create_fuse_moe_impl(
    *,
    n_routed_experts: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    quant_method: QuantizationMethod,
    expert_parallel_state: ExpertParallelState | None = None,
):
    if quant_method.method_name == "mxfp4w4a16-b32-marlin":
        if expert_parallel_state is not None:
            raise RuntimeError("mxfp4w4a16-b32-marlin does not support enable_ep_moe yet")
        return FuseMoeMXFP4(
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            quant_method=quant_method,
        )
    if expert_parallel_state is not None:
        return FuseMoeDeepGEMM(
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            quant_method=quant_method,
            expert_parallel_state=expert_parallel_state,
        )
    if quant_method.method_name == "awq_marlin":
        return FuseMoeMarlin(
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            quant_method=quant_method,
        )
    return FuseMoeTriton(
        n_routed_experts=n_routed_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        quant_method=quant_method,
    )
