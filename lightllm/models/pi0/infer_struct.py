from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Pi0ActionInferStateInfo:
    """Packed action state over a consumer-owned dense Prefix KV snapshot."""

    batch_size: int
    suffix_length: int
    prefix_kv: object
    prefix_lengths: torch.Tensor
    position_cos: torch.Tensor
    position_sin: torch.Tensor
    state: torch.Tensor | None
    noise: torch.Tensor
    action_horizon: int
    time_embeddings: torch.Tensor
    dt: torch.Tensor
    first_suffix_token_isolated: bool
    # Explicitly supplied by the isolated action worker. Keeping this on the
    # infer state makes TP collectives independent of process-global lookup.
    dist_group: object | None = None
    prefill_att_state: object | None = None

    # Reused Gemma projection/FFN helpers inspect these two fields. Action
    # workers do ordinary TP and never enter target prefill balancing.
    need_dp_prefill_balance: bool = False

    def copy_for_cuda_graph(self, source: "Pi0ActionInferStateInfo") -> None:
        if (self.state is None) != (source.state is None):
            raise ValueError("CUDA graph state presence does not match captured input")
        self.noise.copy_(source.noise)
        if self.state is not None:
            self.state.copy_(source.state)
        self.dist_group = source.dist_group
        self.prefix_kv.copy_from(source.prefix_kv)
        self.prefix_lengths.copy_(source.prefix_lengths)
        self.position_cos.copy_(source.position_cos)
        self.position_sin.copy_(source.position_sin)
