from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import triton

from .kernels import _hash, _history


@dataclass(frozen=True)
class NGramHasher:
    multipliers: torch.Tensor
    head_vocab_sizes: torch.Tensor
    head_offsets: torch.Tensor
    eos_token_id: int

    def __post_init__(self):
        if self.multipliers.ndim != 1 or self.multipliers.numel() < 2:
            raise ValueError("N-gram hashing requires at least two multipliers")
        heads = self.head_vocab_sizes.numel()
        if heads == 0 or heads % self.context_len or self.head_vocab_sizes.shape != (heads,):
            raise ValueError("Head count must be a positive multiple of ngram_size - 1")
        if self.head_offsets.shape != (heads,):
            raise ValueError("Head offsets must match head vocabulary sizes")
        for tensor in (self.multipliers, self.head_vocab_sizes, self.head_offsets):
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.dtype != torch.int64:
                raise ValueError("Hash buffers must be contiguous CUDA int64 tensors")
            if tensor.device != self.multipliers.device:
                raise ValueError("Hash buffers must share a device")

        sizes = self.head_vocab_sizes.tolist()
        offsets = self.head_offsets.tolist()
        previous_end = 0
        for size, offset in zip(sizes, offsets):
            if size <= 0 or offset < previous_end or offset + size > torch.iinfo(torch.int64).max:
                raise ValueError("Head vocabularies must have positive sizes and non-overlapping ordered offsets")
            previous_end = offset + size
        if not 0 <= self.eos_token_id <= torch.iinfo(torch.int64).max:
            raise ValueError("EOS token ID must be a nonnegative int64 value")

    @property
    def context_len(self) -> int:
        return self.multipliers.numel() - 1

    @property
    def heads(self) -> int:
        return self.head_vocab_sizes.numel()

    @classmethod
    def from_checkpoint(cls, weights: Mapping[str, torch.Tensor], prefix: str, eos_token_id: int) -> "NGramHasher":
        return cls(
            weights[f"{prefix}.layer_multipliers"],
            weights[f"{prefix}.ngram_heads_vocab_sizes"],
            weights[f"{prefix}.ngram_heads_offsets"],
            eos_token_id,
        )

    def empty_history(self, requests: int) -> torch.Tensor:
        return torch.full(
            (requests, self.context_len), self.eos_token_id, dtype=torch.int64, device=self.multipliers.device
        )

    def _validate(self, tokens: torch.Tensor, starts: torch.Tensor, history: torch.Tensor):
        if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("Tokens must be a packed int32 or int64 vector")
        if starts.ndim != 1 or starts.numel() == 0 or starts.dtype != torch.int32:
            raise ValueError("Starts must be an int32 cumulative-length vector of length requests + 1")
        requests = starts.numel() - 1
        if history.shape != (requests, self.context_len) or history.dtype != torch.int64:
            raise ValueError("History must have shape (requests, ngram_size - 1) and dtype int64")
        if requests == 0 and tokens.numel():
            raise ValueError("Nonempty tokens require at least one request")
        for tensor in (tokens, starts, history):
            if tensor.device != self.multipliers.device or not tensor.is_contiguous():
                raise ValueError("Inputs must be contiguous and on the hash buffers' device")

    @torch.no_grad()
    def hash(self, tokens: torch.Tensor, starts: torch.Tensor, history: torch.Tensor):
        self._validate(tokens, starts, history)
        requests = starts.numel() - 1
        indices = torch.empty((tokens.numel(), self.heads), dtype=torch.int64, device=tokens.device)
        if tokens.numel():
            _hash[(triton.cdiv(tokens.numel(), 64),)](
                tokens,
                starts,
                history,
                self.multipliers,
                self.head_vocab_sizes,
                self.head_offsets,
                indices,
                tokens.numel(),
                requests,
                requests.bit_length(),
                self.context_len,
                self.heads,
                self.heads // self.context_len,
                self.eos_token_id,
                64,
                triton.next_power_of_2(self.heads),
            )
        return indices, self.history_after(tokens, starts, history)

    @torch.no_grad()
    def history_after(
        self,
        tokens: torch.Tensor,
        starts: torch.Tensor,
        history: torch.Tensor,
        accepted_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate(tokens, starts, history)
        requests = starts.numel() - 1
        if accepted_lengths is not None:
            if accepted_lengths.shape != (requests,) or accepted_lengths.dtype != torch.int32:
                raise ValueError("Accepted lengths must be an int32 vector with one count per request")
            if accepted_lengths.device != tokens.device or not accepted_lengths.is_contiguous():
                raise ValueError("Accepted lengths must be contiguous and on the input device")
        output = torch.empty_like(history)
        if requests:
            _history[(requests,)](
                tokens,
                starts,
                history,
                accepted_lengths,
                output,
                self.context_len,
                accepted_lengths is not None,
                self.eos_token_id,
                triton.next_power_of_2(self.context_len),
                num_warps=1,
            )
        return output
