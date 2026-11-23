"""Attention backends and the inputs to build and forward.

Each type is named for how long it lives:

- LayerConfig: one layer's static attention configuration, fixed when the model loads. An
  AttentionBackend instance holds this config.
- AttentionMetadata: request layout and KV cache indices for one forward step.
- AttState: the attention state shared by compatible layers in one microbatch. Backends that
  prepare page tables, recurrent indices or wrappers keep them on their state.
- LayerInputs: one layer call's tensors.

A model passes AttentionBuildResources to create_build_context on first use. Layers whose
backend class and build_context_key match share a context and one state per microbatch.

Attention states are built every step, including CUDA graph replay steps. The caller writes the KV cache
before attention and passes the attention state explicitly rather than through a global forward context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

import torch


LayerConfig = TypeVar("LayerConfig")
AttState = TypeVar("AttState")
LayerInputs = TypeVar("LayerInputs")


class TensorAllocator(Protocol):
    """Allocate contiguous scratch with the requested shape, dtype and device.

    Calls pass a tuple/list shape positionally and dtype/device by keyword,
    as supported by torch.empty and the serving allocator. Returned storage
    must not overlap live operands, outputs or other scratch. The caller's
    allocator manages reuse and CUDA graph lifetimes; the backend retains none.
    """

    def __call__(
        self, shape: tuple[int, ...] | list[int], /, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class AttentionMetadata:
    """Request layout and KV cache indices shared by attention backends for one forward step.

    The caller builds it once per step. Tensors are caller-owned int32/int64 vectors on the
    attention device. req_to_token_indexs[b_req_idx[i], pos] is the physical cache slot of
    request i's token at pos. Under CUDA graphs the caller rewrites the captured step's tensors
    in place before replay, so kernels may read these tensors directly.
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    # Total tokens per request, including any cached prefix and the current tokens.
    b_seq_len: torch.Tensor
    batch_size: int
    # Actual maximum and sum of b_seq_len for this step.
    max_kv_seq_len: int
    total_token_num: int
    # Cumulative query and kv token offsets per request, [batch_size + 1] each.
    cu_q_seq_len: Optional[torch.Tensor] = None
    cu_kv_seq_len: Optional[torch.Tensor] = None
    # Prefill only: start of each request's new queries in packed q and cached prefix lengths.
    b_q_start_loc: Optional[torch.Tensor] = None
    b_ready_cache_len: Optional[torch.Tensor] = None
    max_q_seq_len: int = 1
    # Prefill only: the longest cached prefix and the cache slots of the step's new tokens, in q order.
    max_ready_cache_len: int = 0
    mem_index: Optional[torch.Tensor] = None
    # Prefill only, for layers that attend bidirectionally inside image spans: per query token, the
    # exclusive end of its image span in the request's kv positions, or 0 for a text token, and the
    # span's start (read only for image tokens; needed with a sliding window).
    b_image_token_end: Optional[torch.Tensor] = None
    b_image_token_start: Optional[torch.Tensor] = None
    # Decode query layout. Normally each request has one query row. Speculative decoding gives
    # each logical request decode_rows_per_request consecutive rows that share its KV, the last
    # row seeing the longest prefix. With dynamic_decode_rows the row count varies per request,
    # and consecutive rows with the same b_req_idx form one request; rows of pad_req_idx are
    # padding and each forms its own request.
    decode_rows_per_request: int = 1
    dynamic_decode_rows: bool = False
    pad_req_idx: Optional[int] = None
    # Dynamic speculative rows: each row's draft position within its request. With recurrent
    # state, req_to_mtp_state_index[slot] is the request's last accepted draft position.
    b_mtp_index: Optional[torch.Tensor] = None
    req_to_mtp_state_index: Optional[torch.Tensor] = None
    # Diverse sampling: rows that share a radix-tree prefix node form a group; b_shared_seq_len[i]
    # is how many leading tokens row i shares with its group.
    b_shared_seq_len: Optional[torch.Tensor] = None
    b_shared_radix_node_id: Optional[torch.Tensor] = None
    # Graph capacity for kv lengths when this step is captured or replayed; None in eager mode.
    cuda_graph_max_kv_seq_len: Optional[int] = None
    # Distinguishes the concurrent steps of microbatch overlap.
    microbatch_index: int = 0

    @property
    def uses_cuda_graph(self) -> bool:
        return self.cuda_graph_max_kv_seq_len is not None


@dataclass(frozen=True)
class AttentionBuildResources:
    """Model-wide capacities and workspace access used to initialize a backend's storage."""

    # CUDA device index of the step's tensors.
    device: int
    # Cache slots per page of the KV cache; slot s lies on page s // page_size.
    page_size: int
    # The longest request in tokens, including any speculative-decoding headroom.
    max_seq_len: int
    # Decode rows of the largest CUDA graph step.
    max_cuda_graph_batch_size: int
    # Decode rows any step may have (running requests times rows per request, padded as the
    # model pads its batches); at least max_cuda_graph_batch_size.
    max_decode_rows: int
    # Concurrent steps of microbatch overlap; AttentionMetadata.microbatch_index is below it.
    microbatch_count: int
    # Size of the scratch workspace that implementations needing one share by name.
    workspace_bytes: int
    # get_workspace(name, numel, dtype) returns process-wide storage on device, the same
    # tensor for the same arguments, so builders of one implementation share it.
    get_workspace: Callable[[str, int, torch.dtype], torch.Tensor]
    # Decode batches carry the shared-prefix groups of diverse sampling
    # (AttentionMetadata.b_shared_seq_len and b_shared_radix_node_id).
    diverse_decode: bool = False


class AttentionBackend(ABC, Generic[LayerConfig, AttState, LayerInputs]):
    """Prepare shared state once per step and run attention with each layer's operands."""

    def __init__(self, layer_config: LayerConfig) -> None:
        self.layer_config = layer_config

    @cached_property
    def state_key(self) -> tuple:
        return type(self), self.build_context_key(self.layer_config)

    @abstractmethod
    def forward(
        self,
        att_state: AttState,
        layer_inputs: LayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out itself; concrete impls define tensor layouts.

        Kernels write into out directly. Compute elsewhere and copy only where a kernel cannot: it has
        no out parameter, it needs padded heads, or partial results merge at higher precision first.
        """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @classmethod
    @abstractmethod
    def build(cls, metadata: AttentionMetadata, context: Any) -> AttState:
        """Prepare state once per compatible set of layers before capture or replay.

        Replay reads captured addresses and launch parameters. Keep graph storage for each
        (microbatch, batch size), update its contents in place and size launches for graph capacity.
        """
        pass

    @classmethod
    def create_build_context(cls, layer_config: LayerConfig, ctx: AttentionBuildResources) -> Any:
        """Optionally create backend-owned storage shared across steps with the same build_context_key."""
        return None

    @classmethod
    def build_context_key(cls, layer_config: Any) -> Any:
        """Return the layer fields that affect build, or None for configuration-independent builds."""
        return None

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        """Whether its kernels must not overlap CPU cache copies on the GPU."""
        return False
