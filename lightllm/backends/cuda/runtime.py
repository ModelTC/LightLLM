"""CUDA runtime adapter. Arguments and native object ownership pass through.

Keep calls dynamic so torchada and existing instrumentation of torch.cuda keep
working. No availability probe, CPU fallback or implicit synchronization is
introduced by creating this adapter.
"""

import torch


class CudaRuntime:
    name = "cuda"
    device_type = "cuda"
    device = "cuda"
    distributed_backend = "nccl"
    supports_graphs = True
    supports_tensor_cache = True
    supports_pinned_memory = True
    layer_submission_hook = None

    def device_at(self, index):
        return f"cuda:{index}"

    def move(self, value, *args, **kwargs):
        return value.cuda(*args, **kwargs)

    def is_device_tensor(self, value):
        return value.is_cuda

    def generator(self, *args, **kwargs):
        return torch.Generator(*args, **kwargs)

    def exponential_(self, tensor, generator=None):
        return tensor.exponential_(generator=generator)

    def is_available(self, *args, **kwargs):
        return torch.cuda.is_available(*args, **kwargs)

    def current_device(self, *args, **kwargs):
        return torch.cuda.current_device(*args, **kwargs)

    def set_device(self, *args, **kwargs):
        return torch.cuda.set_device(*args, **kwargs)

    def device_count(self, *args, **kwargs):
        return torch.cuda.device_count(*args, **kwargs)

    def get_device_name(self, *args, **kwargs):
        return torch.cuda.get_device_name(*args, **kwargs)

    def get_device_capability(self, *args, **kwargs):
        return torch.cuda.get_device_capability(*args, **kwargs)

    def get_device_properties(self, *args, **kwargs):
        return torch.cuda.get_device_properties(*args, **kwargs)

    def synchronize(self, *args, **kwargs):
        return torch.cuda.synchronize(*args, **kwargs)

    def current_stream(self, *args, **kwargs):
        return torch.cuda.current_stream(*args, **kwargs)

    def stream_scope(self, *args, **kwargs):
        return torch.cuda.stream(*args, **kwargs)

    def new_stream(self, *args, **kwargs):
        return torch.cuda.Stream(*args, **kwargs)

    def new_event(self, *args, **kwargs):
        return torch.cuda.Event(*args, **kwargs)

    def new_graph(self, *args, **kwargs):
        return torch.cuda.CUDAGraph(*args, **kwargs)

    def capture_graph(self, *args, **kwargs):
        return torch.cuda.graph(*args, **kwargs)

    def graph_pool_handle(self, *args, **kwargs):
        return torch.cuda.graph_pool_handle(*args, **kwargs)

    def is_current_stream_capturing(self, *args, **kwargs):
        return torch.cuda.is_current_stream_capturing(*args, **kwargs)

    def empty_cache(self, *args, **kwargs):
        return torch.cuda.empty_cache(*args, **kwargs)

    def memory_allocated(self, *args, **kwargs):
        return torch.cuda.memory_allocated(*args, **kwargs)

    def memory_reserved(self, *args, **kwargs):
        return torch.cuda.memory_reserved(*args, **kwargs)

    def max_memory_reserved(self, *args, **kwargs):
        return torch.cuda.max_memory_reserved(*args, **kwargs)

    def reset_peak_memory_stats(self, *args, **kwargs):
        return torch.cuda.reset_peak_memory_stats(*args, **kwargs)

    def mem_get_info(self, *args, **kwargs):
        return torch.cuda.mem_get_info(*args, **kwargs)

    def ipc_collect(self, *args, **kwargs):
        return torch.cuda.ipc_collect(*args, **kwargs)

    def manual_seed_all(self, *args, **kwargs):
        return torch.cuda.manual_seed_all(*args, **kwargs)

    def manual_seed(self, *args, **kwargs):
        return torch.cuda.manual_seed(*args, **kwargs)

    def is_bf16_supported(self, *args, **kwargs):
        return torch.cuda.is_bf16_supported(*args, **kwargs)

    def init(self, *args, **kwargs):
        return torch.cuda.init(*args, **kwargs)

    def is_initialized(self, *args, **kwargs):
        return torch.cuda.is_initialized(*args, **kwargs)

    def lazy_init(self, *args, **kwargs):
        return torch.cuda._lazy_init(*args, **kwargs)

    def device_scope(self, *args, **kwargs):
        return torch.cuda.device(*args, **kwargs)

    def range_start(self, *args, **kwargs):
        return torch.cuda.nvtx.range_start(*args, **kwargs)

    def range_end(self, *args, **kwargs):
        return torch.cuda.nvtx.range_end(*args, **kwargs)

    @property
    def event_type(self):
        return torch.cuda.Event

    @property
    def stream_type(self):
        return torch.cuda.Stream

    @property
    def graph_type(self):
        return torch.cuda.CUDAGraph

    @property
    def graph_context_type(self):
        return torch.cuda.graph

    def empty(self, *args, **kwargs):
        return torch.empty(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return torch.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return torch.ones(*args, **kwargs)

    def full(self, *args, **kwargs):
        return torch.full(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **kwargs)

    def tensor(self, *args, **kwargs):
        return torch.tensor(*args, **kwargs)

    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def rand(self, *args, **kwargs):
        return torch.rand(*args, **kwargs)

    def empty_like(self, *args, **kwargs):
        return torch.empty_like(*args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        return torch.zeros_like(*args, **kwargs)

    def ones_like(self, *args, **kwargs):
        return torch.ones_like(*args, **kwargs)

    def full_like(self, *args, **kwargs):
        return torch.full_like(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)

    def rand_like(self, *args, **kwargs):
        return torch.rand_like(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return torch.eye(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return torch.linspace(*args, **kwargs)

    def logspace(self, *args, **kwargs):
        return torch.logspace(*args, **kwargs)

    def as_tensor(self, *args, **kwargs):
        return torch.as_tensor(*args, **kwargs)
