"""Native FP8 FlashMLA implementation of V4 sparse attention."""

import torch
import flash_mla as flashmla
from typing import TYPE_CHECKING, Optional, Tuple

from .dsv4 import DeepseekV4SparseAttBackend, AttControl

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta


class DeepseekV4FlashMlaFp8SparseAttBackend(DeepseekV4SparseAttBackend):
    def create_scheduler_metadata(self):
        return {ratio: flashmla.get_mla_metadata()[0] for ratio in self.compress_ratios}

    def _sparse_attention(
        self,
        q: torch.Tensor,
        packed_kv: Tuple[torch.Tensor, Optional[torch.Tensor]],
        nsa_dict: dict,
        sched_meta: "FlashMLASchedMeta",
        out: torch.Tensor,
        workspace_index: int = 0,
    ) -> torch.Tensor:
        """Attend to SWA plus optional compressed KV and return all padded Q heads."""
        swa_packed, compressed_packed = packed_kv
        # 原子配置提供计算视图；backend 不再计算编码字节数或解释压缩块布局。
        compressed_cache = None
        if compressed_packed is not None:
            config = self.compressed_configs[nsa_dict["compress_ratio"]]
            compressed_cache = config.get_attention_view(packed_kv=compressed_packed)
        swa_cache = self.swa_config.get_attention_view(packed_kv=swa_packed)

        padded_out, _ = flashmla.flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=nsa_dict["head_dim_v"],
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=nsa_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=nsa_dict["swa_indices"],
            attn_sink=nsa_dict["attn_sink"],
            topk_length=nsa_dict["swa_lengths"],
            extra_k_cache=compressed_cache,
            extra_indices_in_kvcache=nsa_dict.get("extra_indices"),
            extra_topk_length=nsa_dict.get("extra_lengths"),
            out=out.unsqueeze(1),
        )
        return padded_out[:, 0]


DSV4_NSA_BACKENDS = {
    "fp8kv_dsa": {"flashmla_sparse": "nsa.dsv4_fp8_flashmla_sparse:DeepseekV4FlashMlaFp8SparseAttBackend"},
    "nvfp4kv": {"triton": "nsa.dsv4_nvfp4:DeepseekV4Nvfp4SparseAttBackend"},
}
DSV4_NSA_BACKENDS["None"] = DSV4_NSA_BACKENDS["fp8kv_dsa"]
