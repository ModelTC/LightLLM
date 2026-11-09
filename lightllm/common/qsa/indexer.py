from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
import triton

from lightllm.common.gated_residual.kernels import grouped_norm
from .indexer_kernels import _rotate, _pool, _scores, _expand
from .topk import cuda_topk_indices, cuda_topk_supported


@dataclass(frozen=True)
class QSAIndexer:
    projection: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    query_heads: int = 4
    token_budget: int = 2048
    compress_ratio: int = 4
    eps: float = 1e-6
    use_cuda_topk: bool = False

    def __post_init__(self):
        dim = self.key_norm.numel()
        if not dim or self.query_heads < 1 or self.compress_ratio < 1 or self.token_budget < 1 or self.eps <= 0:
            raise ValueError("QSA dimensions, budget, compression ratio and epsilon must be positive")
        if self.token_budget % self.compress_ratio:
            raise ValueError("QSA token budget must be divisible by compression ratio")
        if self.key_norm.shape != (dim,) or self.query_norm.shape != (dim,):
            raise ValueError("QSA normalization weights must be matching vectors")
        if (
            self.projection.ndim != 2
            or self.projection.shape[0] != (self.query_heads + 1) * dim
            or not self.projection.shape[1]
        ):
            raise ValueError("QSA projection must contain query heads and one raw key head")
        for weight in (self.projection, self.query_norm, self.key_norm):
            self._activation(weight)

    @property
    def head_dim(self) -> int:
        return self.key_norm.numel()

    @classmethod
    def from_checkpoint(cls, weights: Mapping[str, torch.Tensor], prefix: str, **kwargs) -> "QSAIndexer":
        return cls(
            weights[f"{prefix}.index_qk_proj.weight"],
            weights[f"{prefix}.q_layernorm.weight"],
            weights[f"{prefix}.k_layernorm.weight"],
            **kwargs,
        )

    def _activation(self, x: torch.Tensor):
        if (
            not x.is_cuda
            or not x.is_contiguous()
            or x.device != self.projection.device
            or x.dtype != self.projection.dtype
        ):
            raise ValueError("QSA activations/weights must be contiguous CUDA tensors on the same device and dtype")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("QSA requires floating point activations")

    def _integer(self, x: torch.Tensor):
        if x.dtype != torch.int32 or x.device != self.projection.device or not x.is_contiguous():
            raise ValueError("QSA metadata must be contiguous int32 tensors on the input device")

    def _rotary(self, cos: torch.Tensor, sin: torch.Tensor, rows: int) -> int:
        self._activation(cos)
        self._activation(sin)
        if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != rows:
            raise ValueError("QSA rotary tensors must have matching shapes (tokens, rotary dimensions)")
        if cos.shape[1] > self.head_dim or cos.shape[1] % 2:
            raise ValueError("QSA rotary dimensions must be even and fit the index head")
        return cos.shape[1]

    @torch.no_grad()
    def project(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        self._activation(hidden)
        if hidden.ndim != 2 or hidden.shape[1] != self.projection.shape[1]:
            raise ValueError("QSA hidden input must have shape (tokens, hidden size)")
        rotary_dim = self._rotary(cos, sin, hidden.shape[0])
        projected = F.linear(hidden, self.projection)
        query = projected[:, : self.query_heads * self.head_dim].contiguous().view(-1, self.head_dim)
        query = grouped_norm(query, self.query_norm, 1, self.eps).view(hidden.shape[0], self.query_heads, self.head_dim)
        output = torch.empty_like(query)
        if hidden.shape[0]:
            _rotate[(hidden.shape[0], self.query_heads)](
                query,
                cos,
                sin,
                output,
                self.query_heads,
                self.head_dim,
                rotary_dim,
                triton.next_power_of_2(self.head_dim),
                enable_fp_fusion=False,
            )
        # Keep the raw-key projection as a view.  The state writer accepts a
        # strided source, so making this slice contiguous only adds a full
        # token-by-head-dimension copy on every QSA layer.
        return output, projected[:, self.query_heads * self.head_dim :]

    def empty_cache(self, requests: int, max_tokens: int) -> torch.Tensor:
        return torch.empty(
            (requests, max_tokens // self.compress_ratio, self.head_dim),
            dtype=self.projection.dtype,
            device=self.projection.device,
        )

    def empty_physical_cache(self, slots: int) -> torch.Tensor:
        if slots < 0:
            raise ValueError("QSA physical cache size cannot be negative")
        return torch.empty((slots, self.head_dim), dtype=self.projection.dtype, device=self.projection.device)

    @torch.no_grad()
    def pool(
        self,
        raw_keys: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: torch.Tensor,
        first_blocks: torch.Tensor,
        max_new_blocks: int,
        request_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._activation(raw_keys)
        self._activation(cache)
        for tensor in (indices, lengths, first_blocks):
            self._integer(tensor)
        if raw_keys.ndim != 2 or raw_keys.shape[1] != self.head_dim or indices.ndim != 2:
            raise ValueError("QSA raw keys and slot indices must be matrices")
        requests, capacity = indices.shape
        active = requests
        if request_rows is not None:
            self._integer(request_rows)
            if request_rows.ndim != 1:
                raise ValueError("QSA active request rows must be a vector")
            active = request_rows.numel()
        if lengths.shape != (active,) or first_blocks.shape != (active,):
            raise ValueError("QSA lengths and first blocks must have one entry per request")
        physical = cache.ndim == 2
        if physical:
            if cache.shape != raw_keys.shape:
                raise ValueError("QSA physical pooled cache must match the raw key namespace")
        elif (
            cache.ndim != 3
            or cache.shape[0] != requests
            or cache.shape[1] < capacity // self.compress_ratio
            or cache.shape[2] != self.head_dim
        ):
            raise ValueError("QSA cache must hold all complete blocks in each request's slot table")
        if max_new_blocks < 0:
            raise ValueError("QSA maximum new block count cannot be negative")
        rotary_dim = self._rotary(cos, sin, raw_keys.shape[0])
        if active and max_new_blocks:
            _pool[(active, max_new_blocks)](
                raw_keys,
                indices,
                lengths,
                first_blocks,
                self.key_norm,
                cos,
                sin,
                cache,
                request_rows,
                request_rows is not None,
                physical,
                capacity,
                cache.shape[1],
                self.head_dim,
                rotary_dim,
                self.compress_ratio,
                self.eps,
                triton.next_power_of_2(self.head_dim),
                triton.next_power_of_2(self.compress_ratio),
                enable_fp_fusion=False,
            )
        return cache

    @torch.no_grad()
    def select(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        requests: torch.Tensor,
        visible_lengths: torch.Tensor,
        indices: torch.Tensor,
        query_chunk_size: int = 64,
    ) -> torch.Tensor:
        self._activation(query)
        self._activation(cache)
        for tensor in (requests, visible_lengths, indices):
            self._integer(tensor)
        if query.ndim != 3 or query.shape[1:] != (self.query_heads, self.head_dim):
            raise ValueError("QSA query shape must be (tokens, index query heads, index head dimension)")
        tokens = query.shape[0]
        if requests.shape != (tokens,) or visible_lengths.shape != (tokens,) or indices.ndim != 2:
            raise ValueError("QSA requires one request ID and visible length per query")
        physical = cache.ndim == 2
        if physical:
            if cache.shape[1] != self.head_dim:
                raise ValueError("QSA physical pooled cache has the wrong head dimension")
        elif (
            cache.ndim != 3
            or cache.shape[0] != indices.shape[0]
            or cache.shape[2] != self.head_dim
            or cache.shape[1] < indices.shape[1] // self.compress_ratio
        ):
            raise ValueError("QSA pooled cache and physical slot table dimensions do not match")
        if query_chunk_size < 1:
            raise ValueError("QSA query chunk size must be positive")
        width = self.token_budget + self.compress_ratio - 1
        output = torch.empty((tokens, width), dtype=torch.int32, device=query.device)
        blocks = min(indices.shape[1], cache.shape[0]) // self.compress_ratio if physical else cache.shape[1]
        block_budget = self.token_budget // self.compress_ratio
        topk = min(blocks, block_budget)
        cuda_topk = self.use_cuda_topk and cuda_topk_supported(query.device, blocks, topk)
        max_score_elements = 16 * 1024 * 1024
        query_chunk_size = min(query_chunk_size, max(1, max_score_elements // max(blocks, 1)))
        score_buffer = torch.empty((min(tokens, query_chunk_size), blocks), dtype=torch.float32, device=query.device)
        for first in range(0, tokens, query_chunk_size):
            count = min(query_chunk_size, tokens - first)
            scores = score_buffer[:count]
            if blocks:
                _scores[(count, triton.cdiv(blocks, 64))](
                    query,
                    cache,
                    requests,
                    visible_lengths,
                    indices,
                    scores,
                    first,
                    self.query_heads,
                    self.head_dim,
                    blocks,
                    indices.shape[1],
                    physical,
                    self.compress_ratio,
                    max(16, triton.next_power_of_2(self.query_heads)),
                    max(16, triton.next_power_of_2(self.head_dim)),
                    64,
                    block_budget if cuda_topk else 0,
                    "ieee" if query.dtype == torch.float32 else "tf32",
                )
            selected = (
                cuda_topk_indices(
                    scores, topk, visible_lengths=visible_lengths, first=first, compress_ratio=self.compress_ratio
                )
                if cuda_topk
                else torch.topk(scores, topk, dim=-1, sorted=False).indices
            )
            _expand[(count,)](
                selected,
                requests,
                visible_lengths,
                indices,
                output,
                first,
                topk,
                block_budget,
                indices.shape[1],
                self.compress_ratio,
                width,
                triton.next_power_of_2(width),
            )
        return output
