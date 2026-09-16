from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .triton_ep_impl import FuseMoeTritonEP
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
    ep_moe_backend: str = "auto",
):
    if quant_method.method_name == "mxfp4w4a16-b32-marlin":
        if expert_parallel_state is not None:
            raise RuntimeError("mxfp4w4a16-b32-marlin does not support enable_ep_moe yet")
        impl_cls = FuseMoeMXFP4
    elif expert_parallel_state is not None:
        use_triton_ep = ep_moe_backend == "triton" and quant_method.method_name == "fp8w8a8-b128-deepgemm"
        impl_cls = FuseMoeTritonEP if use_triton_ep else FuseMoeDeepGEMM
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
