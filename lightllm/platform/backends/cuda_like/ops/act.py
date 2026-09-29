from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.platform.ops.act import ActOps


CUDA_LIKE_ACT_OPS = ActOps(
    silu_and_mul=silu_and_mul_fwd,
    gelu_and_mul=gelu_and_mul_fwd,
)
