from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.layernorm import layernorm_forward
from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward, qk_rmsnorm_fused_forward
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.platform.ops.norm import NormOps


CUDA_LIKE_NORM_OPS = NormOps(
    rms_norm=rmsnorm_forward,
    gated_rms_norm=gated_rmsnorm_forward,
    layer_norm=layernorm_forward,
    qk_rms_norm=qk_rmsnorm_forward,
    fused_qk_rms_norm=qk_rmsnorm_fused_forward,
)
