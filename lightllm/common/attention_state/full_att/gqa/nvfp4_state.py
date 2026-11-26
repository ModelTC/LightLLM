"""Dynamic block-16 E2M1 KV, with data and scale bytes in the same token row.

NVFP4 follows the vLLM/FlashInfer recipe with unit global K/V scales, the
fallback used when a checkpoint does not contain calibrated global scales.
"""

from dataclasses import dataclass

import torch

from .gqa_state import GQAAttentionConfig, GQAFullAttentionState


@dataclass(frozen=True)
class NVFP4AttentionConfig(GQAAttentionConfig):
    def __post_init__(self):
        super().__post_init__()
        assert self.head_dim in (16, 32, 64, 128, 256), "NVFP4 supports head dimensions 16 through 256"
        assert self.dtype in (torch.float16, torch.bfloat16), "NVFP4 requires FP16/BF16 inputs"


class NVFP4FullAttentionState(GQAFullAttentionState):
    def __init__(self, layout_config, page_size=1, device="cuda"):
        super().__init__(layout_config, page_size=page_size, device=device)
        self.storage_dtype = torch.uint8

    def get_state_type(self):
        return f"gqa_nvfp4_h{self.config.global_head_num}_d{self.config.head_dim}"

    def get_token_bytes(self):
        return len(self.config.logical_layer_groups) * 2 * self.config.head_num * self._storage_shape()[-1]

    def _storage_shape(self):
        # Trailing bytes are E4M3 scales, one per 16 values. Keeping them
        # inside the row makes every existing copy/CPU/DP/PD transfer atomic
        # with respect to the logical token's complete encoding.
        return (
            len(self.config.logical_layer_groups),
            self.capacity + self.page_size,
            self.config.head_num,
            self.config.head_dim // 2 + self.config.head_dim // 16,
        )

    def write_full_att_kv(self, logical_layer_id, token_indices, kv):
        from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4 import quantize_nvfp4_kv

        row = self.get_storage_layer_index(logical_layer_id)
        quantize_nvfp4_kv(
            kv,
            token_indices,
            self.k_storage[row],
            self.v_storage[row],
        )
