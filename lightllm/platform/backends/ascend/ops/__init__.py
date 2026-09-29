from .act import ASCEND_ACT_OPS
from .norm import ASCEND_NORM_OPS
from .sampling import ASCEND_SAMPLING_OPS
from lightllm.platform.ops import PlatformOps


ASCEND_OPS = PlatformOps(
    norm=ASCEND_NORM_OPS,
    act=ASCEND_ACT_OPS,
    sampling=ASCEND_SAMPLING_OPS,
)

__all__ = ["ASCEND_OPS"]
