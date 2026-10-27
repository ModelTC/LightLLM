from lightllm.platform.ops import ActOps

from .common import unsupported_op

ASCEND_ACT_OPS = ActOps(
    silu_and_mul=unsupported_op("act.silu_and_mul"),
    gelu_and_mul=unsupported_op("act.gelu_and_mul"),
)
