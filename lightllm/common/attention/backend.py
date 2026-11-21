"""Attention backend interfaces: the backend, its step builder and its impl.

Each type is named for how long it lives:

- LayerConfig: one layer's static attention configuration, fixed when the model loads. An
  AttentionImpl is created once per layer config.
- AttentionBatch: one forward step's requests, independent of any implementation.
- StepArgs: an implementation's kernel arguments for one step, shared by every layer of a layer
  group. Its AttentionStepBuilder builds them from the batch before the first layer.
- LayerInputs: one layer call's tensors.

Step args are built every step, including CUDA graph replay steps. The caller writes the KV cache
before attention and passes the step args explicitly rather than through a global forward context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar

import torch


LayerConfig = TypeVar("LayerConfig")
StepArgs = TypeVar("StepArgs")
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
class AttentionBatch:
    """The requests of one forward step, independent of any attention implementation.

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


class AttentionStepBuilder(ABC, Generic[StepArgs]):
    """Build an implementation's step args for one step of one layer group.

    build() runs once per step, including CUDA graph capture and replay steps, before the first
    layer and outside capture. All layers of the group share its result. For a CUDA graph step,
    any state derived from the batch must be written into storage the step builder keeps for
    that (microbatch, batch size), so that a replay sees the values of the step being replayed.
    Eager steps may use fresh storage.

    A replay reruns the kernels captured with the capture step's step args, so only tensor
    contents change between replays. Step-args fields read on the host, such as launch sizes or
    kernel configs, stay as captured: a replay step's own step args are not used by the graph,
    and such fields must be valid for every step the graph may replay.
    """

    @abstractmethod
    def build(self, batch: AttentionBatch) -> StepArgs:
        pass


class AttentionImpl(ABC, Generic[LayerConfig, StepArgs, LayerInputs]):
    """Run attention for every layer that shares one layer config."""

    def __init__(self, layer_config: LayerConfig) -> None:
        self.layer_config = layer_config

    @abstractmethod
    def forward(
        self,
        step_args: StepArgs,
        layer_inputs: LayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out itself; concrete impls define tensor layouts."""


class AttentionBackend(ABC):
    """Describe one attention implementation for one family and phase by naming its step builder and impl.

    Not to be confused with lightllm.common.basemodel.attention.base_att.BaseAttBackend, the
    model-facing adapter that builds batches, layer configs and layer inputs and calls step
    builders and impls.
    """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_step_builder_cls() -> type[AttentionStepBuilder]:
        pass

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type[AttentionImpl]:
        pass
