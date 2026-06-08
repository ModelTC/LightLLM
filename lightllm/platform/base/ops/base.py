import inspect
import torch
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable


def get_protocol_op_names(protocol: type) -> tuple[str, ...]:
    """ Get the names of the public methods in the protocol. """
    names: list[str] = []
    for name, value in protocol.__dict__.items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value):
            names.append(name)
    return tuple(names)


@runtime_checkable
class OpsProtocol(Protocol):

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
    ) -> None: ...

    def offload_embed_tensor_to_cache(
        self,
        *,
        embed_tensor: torch.Tensor,
        cache_tensor: torch.Tensor,
        start_index_in_cache: int,
    ) -> None: ...

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
    ) -> None: ...

    def ffn(
        self,
        *,
        input: torch.Tensor,
        layer_weight: Any,
        alloc_func: Callable,
        embed_dim: int,
    ) -> torch.Tensor: ...

    def embedding(
        self,
        *,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
        vob_start_id: int = 0,
        vob_end_id: Optional[int] = None,
    ) -> torch.Tensor: ...

    def lm_head(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor: ...

    def rms_norm(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
        gate_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...

    def layer_norm(
        self,
        *,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out: Optional[torch.Tensor] = None,
        alloc_func: Callable = torch.empty,
    ) -> torch.Tensor: ...

    def qk_rms_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        eps: float,
        fp32_multiply: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


# op names for ops, the order is the same as the declaration order in the protocol
OP_NAMES: tuple[str, ...] = get_protocol_op_names(OpsProtocol)
