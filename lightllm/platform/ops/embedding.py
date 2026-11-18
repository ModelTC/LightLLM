from dataclasses import dataclass
from typing import Protocol

import torch


class Embedding(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        vob_start_id: int,
        vob_end_id: int,
        out: torch.Tensor,
    ) -> None: ...


class MultimodalEmb(Protocol):
    def __call__(
        self,
        out: torch.Tensor,
        prompt_ids: torch.Tensor,
        text_weight_embs: torch.Tensor,
        embed_cache: torch.Tensor,
        img_token_lens: torch.Tensor,
        img_start_token_ids: torch.Tensor,
        img_start_locs_in_cache: torch.Tensor,
        tp_text_start_token_id: int,
        tp_text_end_token_id: int,
        tp_world_size: int,
        text_embed_scale: float = 1.0,
    ) -> None: ...


class OffloadEmbedTensorToCache(Protocol):
    def __call__(
        self,
        embed_tensor: torch.Tensor,
        cache_tensor: torch.Tensor,
        start_index_in_cache: int,
    ) -> None: ...


@dataclass(frozen=True)
class EmbeddingOps:
    embedding: Embedding
    multimodal_emb: MultimodalEmb
    offload_embed_tensor_to_cache: OffloadEmbedTensorToCache
