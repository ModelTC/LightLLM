"""Attention backends and the inputs to prepare and forward.

Each type is named for how long it lives:

- LayerConfig: one layer's static attention configuration, fixed when the model loads. An
  AttentionBackend instance holds this config.
- InferStateInfo: request layout and KV cache indices for one forward step.
- AttState: the attention state shared by compatible layers in one microbatch. Backends that
  prepare page tables, recurrent indices or wrappers keep them on their state.
- LayerInputs: one layer call's tensors.

In each phase, one backend instance prepares state for the layers whose state_key matches: the
backend class plus group_key(layer_config). That instance sets up its storage on first use and
keeps the storage CUDA graphs read.

Attention states are built every step, including CUDA graph replay steps. The caller writes the KV cache
before attention and passes the attention state explicitly rather than through a global forward context.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Protocol, TypeVar

import torch

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo

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
class AttentionBuildResources:
    """Model-wide capacities and workspace access used to set up a backend's storage."""

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
    # Concurrent steps of microbatch overlap; InferStateInfo.microbatch_index is below it.
    microbatch_count: int
    # Size of the scratch workspace that implementations needing one share by name.
    workspace_bytes: int
    # get_workspace(name, numel, dtype) returns process-wide storage on device, the same
    # tensor for the same arguments, so backends of one implementation share it.
    get_workspace: Callable[[str, int, torch.dtype], torch.Tensor]
    # Decode batches carry the shared-prefix groups of diverse sampling
    # (InferStateInfo.b_shared_seq_len and b_shared_radix_node_id).
    diverse_decode: bool = False


class AttentionBackend(ABC, Generic[LayerConfig, AttState, LayerInputs]):
    """Prepare shared state once per step and run attention with each layer's operands."""

    # The backend's name in logs and selection errors.
    name: ClassVar[str]
    # The phase the backend runs in, "prefill" or "decode"; the selector binds it only to that phase.
    phase: ClassVar[str]

    def __init__(self, layer_config: LayerConfig) -> None:
        self._check_supported(layer_config)
        self.layer_config = layer_config
        self._is_set_up = False
        # Tensors CUDA graphs read, kept at fixed addresses per (microbatch, batch size, name).
        self._graph_tensors: dict[tuple, torch.Tensor] = {}

    @classmethod
    def _check_supported(cls, layer_config: LayerConfig) -> None:
        """Raise NotImplementedError if this backend cannot run the layer config."""

    @classmethod
    def group_key(cls, layer_config: LayerConfig) -> Any:
        """The layer config fields that setup and build read; layers with equal keys share one state."""
        return None

    @cached_property
    def state_key(self) -> tuple:
        return type(self), self.group_key(self.layer_config)

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        """Whether its kernels must not overlap CPU cache copies on the GPU."""
        return False

    def prepare(self, infer_state: "InferStateInfo", resources: AttentionBuildResources) -> AttState:
        """Prepare this phase's shared state for one microbatch, setting up storage on first use."""
        if not self._is_set_up:
            self.setup(resources)
            self._is_set_up = True
        return self.build(infer_state)

    def setup(self, resources: AttentionBuildResources) -> None:
        """Size the storage build uses from model-wide resources; called once before the first build."""

    @abstractmethod
    def build(self, infer_state: "InferStateInfo") -> AttState:
        """Prepare state once per compatible set of layers before capture or replay.

        Read only the layer config fields in group_key. Replay reads captured addresses and launch
        parameters: keep graph storage for each (microbatch, batch size), update its contents in
        place and size launches for graph capacity.
        """

    @abstractmethod
    def forward(
        self,
        att_state: AttState,
        layer_inputs: LayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out itself; concrete backends define tensor layouts.

        Kernels write into out directly. Compute elsewhere and copy only where a kernel cannot: it has
        no out parameter, it needs padded heads, or partial results merge at higher precision first.
        """

    def _keep_for_graph(self, infer_state: "InferStateInfo", name: str, value: torch.Tensor) -> torch.Tensor:
        """Return value, or for a CUDA graph step its copy at a fixed address per (microbatch, batch size).

        The first call for a key keeps a clone; later calls copy value into it in place, so tensors a
        captured CUDA graph reads see each replayed step's values.
        """
        if not infer_state.uses_cuda_graph:
            return value
        key = (infer_state.microbatch_index, infer_state.batch_size, name)
        kept = self._graph_tensors.get(key)
        if kept is None:
            kept = self._graph_tensors[key] = value.clone()
        else:
            kept.copy_(value)
        return kept
