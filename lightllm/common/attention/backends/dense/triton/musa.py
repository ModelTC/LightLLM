from lightllm.common.attention.api import AttentionPhase, CapabilityResult
from lightllm.platform import Platform

from .backend import TritonAttBackend


class MusaTritonAttBackend(TritonAttBackend):
    """MUSA Triton 的普通 Dense 路径；其它执行模式须分别移植和验证。"""

    supported_platforms = frozenset({Platform.MUSA})
    supported_graph_phases = frozenset({AttentionPhase.DECODE})
    supports_alibi = False
    supports_sliding_window = False
    supports_expert_parallel = False

    @classmethod
    def check_support(cls, spec, caps=None, execution=None):
        result = super().check_support(spec, caps, execution)
        if not result.supported:
            return result
        if spec.use_mtp or spec.dynamic_verify:
            return CapabilityResult.reject("MUSA Dense does not implement speculative verification")
        if execution is not None and (
            execution.microbatch_count != 1 or execution.transfer_paths or execution.max_verify_tokens != 1
        ):
            return CapabilityResult.reject("MUSA Dense requires local state without microbatch overlap")
        return result
