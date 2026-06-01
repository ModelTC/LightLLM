import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class InferOps(ABC):

    @abstractmethod
    def multimodal_emb(
        self,
        *,
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
    ) -> None:
        pass

    @abstractmethod
    def offload_embed_tensor_to_cache(
        self,
        *,
        embed_tensor: torch.Tensor,
        cache_tensor: torch.Tensor,
        start_index_in_cache: int,
    ) -> None:
        pass

    @abstractmethod
    def rotary_emb(
        self,
        *,
        is_prefill: bool,
        batch_size: int,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        partial_rotary_factor: float = 1.0,
        rotary_impl: Optional[Callable] = None,
    ) -> None:
        pass

    @abstractmethod
    def ffn(
        self,
        *,
        input: torch.Tensor,
        layer_weight: Any,
        alloc_func: Callable,
        embed_dim: int,
    ) -> torch.Tensor:
        pass
