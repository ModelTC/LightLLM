from lightllm.common.attention.api.spec import CacheEncoding, AttentionPhase
import torch

from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNPrefillBackend


class FlashQlaGDNPrefillBackend(TritonGDNPrefillBackend):
    phase = AttentionPhase.PREFILL

    provider = "flashqla"

    dependency = "flash_qla"

    activation_dtypes = frozenset({torch.float16, torch.bfloat16})

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    def get_prefill_kernel(self):
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
