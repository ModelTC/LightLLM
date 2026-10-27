from lightllm.platform.backends.cuda_like.ops.act import CUDA_LIKE_ACT_OPS
from lightllm.platform.backends.cuda_like.ops.norm import CUDA_LIKE_NORM_OPS
from lightllm.platform.backends.cuda_like.ops.sampling import CUDA_LIKE_SAMPLING_OPS
from lightllm.platform.ops import PlatformOps


CUDA_LIKE_OPS = PlatformOps(
    norm=CUDA_LIKE_NORM_OPS,
    act=CUDA_LIKE_ACT_OPS,
    sampling=CUDA_LIKE_SAMPLING_OPS,
)

__all__ = ["CUDA_LIKE_OPS"]
