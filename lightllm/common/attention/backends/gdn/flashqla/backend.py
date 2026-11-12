import torch

from lightllm.common.attention.api import AttentionPhase
from lightllm.common.attention.backends.gdn.triton.backend import TritonLinearAttBackend


class FlashQlaLinearAttBackend(TritonLinearAttBackend):
    provider = "flashqla"
    dependency = "flash_qla"
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supported_phases = frozenset({AttentionPhase.PREFILL})
    cache_formats = frozenset({"gdn"})

    def get_prefill_kernel(self):
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
