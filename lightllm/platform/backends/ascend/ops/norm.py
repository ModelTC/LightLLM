from lightllm.platform.ops import NormOps

from .common import unsupported_op

ASCEND_NORM_OPS = NormOps(
    rms_norm=unsupported_op("norm.rms_norm"),
    gated_rms_norm=unsupported_op("norm.gated_rms_norm"),
    layer_norm=unsupported_op("norm.layer_norm"),
    qk_rms_norm=unsupported_op("norm.qk_rms_norm"),
    fused_qk_rms_norm=unsupported_op("norm.fused_qk_rms_norm"),
)
