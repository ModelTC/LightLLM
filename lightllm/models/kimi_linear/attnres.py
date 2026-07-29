from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class BlockAttnResConfig:
    block_size: int

    @classmethod
    def from_network_config(cls, network_config: Mapping) -> Optional["BlockAttnResConfig"]:
        block_size = network_config.get("attn_res_block_size")
        if block_size is None:
            return None
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("attn_res_block_size must be a positive integer")
        return cls(block_size=block_size)


def _validate_mix_inputs(
    sources: Sequence[torch.Tensor], query: torch.Tensor, norm_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sources:
        raise ValueError("AttnRes needs at least one source")

    first = sources[0]
    if first.ndim != 2:
        raise ValueError(f"AttnRes sources must have shape [tokens, hidden], got {tuple(first.shape)}")
    for source in sources[1:]:
        if source.shape != first.shape:
            raise ValueError("all AttnRes sources must have the same shape")
        if source.dtype != first.dtype or source.device != first.device:
            raise ValueError("all AttnRes sources must have the same dtype and device")

    query = query.reshape(-1)
    norm_weight = norm_weight.reshape(-1)
    hidden_size = first.shape[-1]
    if query.numel() != hidden_size or norm_weight.numel() != hidden_size:
        raise ValueError(
            "AttnRes projection and norm weights must match the source hidden size: "
            f"hidden={hidden_size}, projection={query.numel()}, norm={norm_weight.numel()}"
        )
    if query.device != first.device or norm_weight.device != first.device:
        raise ValueError("AttnRes projection, norm weights, and sources must be on the same device")
    return query, norm_weight


def block_attnres_mix(
    sources: Sequence[torch.Tensor],
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    query, norm_weight = _validate_mix_inputs(sources, query, norm_weight)
    first = sources[0]
    if len(sources) == 1:
        return first

    compute_dtype = torch.float32 if first.dtype in (torch.float16, torch.bfloat16) else first.dtype
    score_weight = query.to(compute_dtype) * norm_weight.to(compute_dtype)
    logits = []
    for source in sources:
        source_fp = source.to(compute_dtype)
        inv_rms = torch.rsqrt(source_fp.square().mean(dim=-1) + eps)
        logits.append((source_fp * score_weight).sum(dim=-1) * inv_rms)

    depth_weights = torch.stack(logits, dim=0).softmax(dim=0)
    output = torch.zeros_like(first, dtype=compute_dtype)
    for source, weight in zip(sources, depth_weights):
        output.add_(source.to(compute_dtype) * weight.unsqueeze(-1))
    return output.to(first.dtype)


def normalize_attnres_query_weight(weight: torch.Tensor, hidden_size: int) -> torch.Tensor:
    if weight.numel() != hidden_size:
        return weight
    return weight.reshape(hidden_size)


@dataclass
class BlockAttnResState:
    block_size: int
    prefix_sum: Optional[torch.Tensor]
    block_residuals: list[torch.Tensor]

    @classmethod
    def from_embedding(cls, embedding: torch.Tensor, block_size: int) -> "BlockAttnResState":
        if block_size <= 0:
            raise ValueError("AttnRes block_size must be positive")
        return cls(block_size=block_size, prefix_sum=embedding, block_residuals=[])

    def sources(self) -> list[torch.Tensor]:
        if self.prefix_sum is None:
            return list(self.block_residuals)
        return [*self.block_residuals, self.prefix_sum]

    def mix(self, query: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
        return block_attnres_mix(self.sources(), query=query, norm_weight=norm_weight, eps=eps)

    def begin_layer(self, layer_num: int) -> None:
        if layer_num % self.block_size != 0:
            return
        if self.prefix_sum is None:
            raise RuntimeError("AttnRes cannot start a block without a prefix sum")
        self.block_residuals.append(self.prefix_sum)
        self.prefix_sum = None

    def add_sublayer_output(self, output: torch.Tensor) -> None:
        reference = self.block_residuals[0] if self.block_residuals else self.prefix_sum
        if reference is None:
            self.prefix_sum = output
            return
        if output.shape != reference.shape:
            raise ValueError("AttnRes sublayer output must match the embedding shape")
        if output.dtype != reference.dtype or output.device != reference.device:
            raise ValueError("AttnRes sublayer output must match the embedding dtype and device")
        self.prefix_sum = output if self.prefix_sum is None else self.prefix_sum + output

    def finish(self, query: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
        return self.mix(query=query, norm_weight=norm_weight, eps=eps)
