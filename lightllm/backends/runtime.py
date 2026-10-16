"""Torch tensor runtime contract; handles retain their native semantics.

No implementation is selected by inspecting the host. CUDA is the explicit
default, including its existing torchada compatibility. Other runtimes must be
registered and selected before importing inference modules.
"""

from typing import Any, Protocol


class DeviceRuntime(Protocol):
    name: str
    device_type: str
    device: str
    supports_graphs: bool
    supports_tensor_cache: bool
    supports_pinned_memory: bool
    distributed_backend: str
    event_type: Any
    stream_type: Any
    graph_type: Any
    graph_context_type: Any
    # Optional enqueue hint between decode layers; never a completion barrier.
    layer_submission_hook: Any

    def device_at(self, index) -> str:
        ...

    def move(self, value, *args, **kwargs):
        ...

    def is_device_tensor(self, value) -> bool:
        ...

    def generator(self, *args, **kwargs):
        ...

    def exponential_(self, tensor, generator=None):
        ...

    def is_available(self, *args, **kwargs):
        ...

    def current_device(self, *args, **kwargs):
        ...

    def set_device(self, *args, **kwargs):
        ...

    def device_count(self, *args, **kwargs):
        ...

    def get_device_name(self, *args, **kwargs):
        ...

    def get_device_capability(self, *args, **kwargs):
        ...

    def get_device_properties(self, *args, **kwargs):
        ...

    def synchronize(self, *args, **kwargs):
        ...

    def current_stream(self, *args, **kwargs):
        ...

    def stream_scope(self, *args, **kwargs):
        ...

    def new_stream(self, *args, **kwargs):
        ...

    def new_event(self, *args, **kwargs):
        ...

    def new_graph(self, *args, **kwargs):
        ...

    def capture_graph(self, *args, **kwargs):
        ...

    def graph_pool_handle(self, *args, **kwargs):
        ...

    def is_current_stream_capturing(self, *args, **kwargs):
        ...

    def empty_cache(self, *args, **kwargs):
        ...

    def memory_allocated(self, *args, **kwargs):
        ...

    def memory_reserved(self, *args, **kwargs):
        ...

    def max_memory_reserved(self, *args, **kwargs):
        ...

    def reset_peak_memory_stats(self, *args, **kwargs):
        ...

    def mem_get_info(self, *args, **kwargs):
        ...

    def ipc_collect(self, *args, **kwargs):
        ...

    def manual_seed_all(self, *args, **kwargs):
        ...

    def manual_seed(self, *args, **kwargs):
        ...

    def is_bf16_supported(self, *args, **kwargs):
        ...

    def init(self, *args, **kwargs):
        ...

    def is_initialized(self, *args, **kwargs):
        ...

    def lazy_init(self, *args, **kwargs):
        ...

    def device_scope(self, *args, **kwargs):
        ...

    def range_start(self, *args, **kwargs):
        ...

    def range_end(self, *args, **kwargs):
        ...

    def empty(self, *args, **kwargs):
        ...

    def zeros(self, *args, **kwargs):
        ...

    def ones(self, *args, **kwargs):
        ...

    def full(self, *args, **kwargs):
        ...

    def arange(self, *args, **kwargs):
        ...

    def tensor(self, *args, **kwargs):
        ...

    def randn(self, *args, **kwargs):
        ...

    def randint(self, *args, **kwargs):
        ...

    def rand(self, *args, **kwargs):
        ...

    def empty_like(self, *args, **kwargs):
        ...

    def zeros_like(self, *args, **kwargs):
        ...

    def ones_like(self, *args, **kwargs):
        ...

    def full_like(self, *args, **kwargs):
        ...

    def randn_like(self, *args, **kwargs):
        ...

    def rand_like(self, *args, **kwargs):
        ...

    def eye(self, *args, **kwargs):
        ...

    def linspace(self, *args, **kwargs):
        ...

    def logspace(self, *args, **kwargs):
        ...

    def as_tensor(self, *args, **kwargs):
        ...
