from contextlib import nullcontext
from dataclasses import dataclass

import torch


class CompletedFence:
    def query(self):
        return True

    def synchronize(self):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass


@dataclass(frozen=True)
class DeviceRuntime:
    device: torch.device

    def __post_init__(self):
        if self.device.type not in ("cpu", "cuda", "meta"):
            raise ValueError(f"no execution runtime for {self.device.type}")
        if self.device.type == "cuda" and self.device.index is None:
            object.__setattr__(self, "device", torch.device("cuda", torch.cuda.current_device()))

    @property
    def tracks_memory_peaks(self):
        return self.device.type == "cuda"

    def allocate(self, shape, dtype, fill=None):
        if fill is None:
            return torch.empty(shape, dtype=dtype, device=self.device)
        return torch.full(shape, fill, dtype=dtype, device=self.device)

    def copy_to(self, tensor, non_blocking=False):
        return tensor.to(device=self.device, non_blocking=non_blocking)

    def stream(self, stream=None):
        if self.device.type == "cpu":
            return nullcontext()
        if self.device.type != "cuda":
            raise ValueError(f"streams are unavailable on {self.device}")
        return torch.cuda.stream(stream)

    def current_stream(self):
        return None if self.device.type == "cpu" else torch.cuda.current_stream(self.device)

    def record_completion(self):
        if self.device.type == "cpu":
            return CompletedFence()
        if self.device.type != "cuda":
            raise ValueError(f"completion events are unavailable on {self.device}")
        event = torch.cuda.Event()
        event.record(self.current_stream())
        return event

    def wait(self, fence):
        if self.device.type == "cpu" or isinstance(fence, CompletedFence):
            fence.synchronize()
        else:
            self.current_stream().wait_event(fence)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def empty_cache(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def memory_info(self):
        if self.device.type == "cpu":
            import psutil

            memory = psutil.virtual_memory()
            return memory.available, memory.total
        return torch.cuda.mem_get_info(self.device)

    def reset_memory_peak(self):
        if self.tracks_memory_peaks:
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats(self):
        if self.device.type == "cpu":
            import psutil

            allocated = psutil.Process().memory_info().rss
            return dict(
                allocated_bytes=allocated,
                reserved_bytes=allocated,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
                device_used_bytes=None,
            )
        free, total = self.memory_info()
        return dict(
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            device_used_bytes=total - free,
        )

    def create_graph(self):
        if self.device.type != "cuda":
            raise ValueError(f"graph capture is unavailable on {self.device}")
        return torch.cuda.CUDAGraph()

    def capture_graph(self, graph, pool=None):
        if self.device.type != "cuda":
            raise ValueError(f"graph capture is unavailable on {self.device}")
        return torch.cuda.graph(graph, pool=pool)

    def graph_runner(self, phase):
        if self.device.type != "cuda":
            raise ValueError(f"{phase} graph execution is unavailable on {self.device}")
        if phase == "decode":
            from .basemodel.cuda_graph import CudaGraph

            return CudaGraph
        if phase == "prefill":
            from .basemodel.prefill_cuda_graph import PrefillCudaGraph

            return PrefillCudaGraph
        raise ValueError(f"unknown graph phase {phase}")


_runtime_factories = {kind: DeviceRuntime for kind in ("cpu", "cuda", "meta")}


def register_device_runtime(device_type, factory):
    if device_type in _runtime_factories:
        raise ValueError(f"runtime already registered for {device_type}")
    _runtime_factories[device_type] = factory


def get_device_runtime(device):
    device = torch.device(device)
    try:
        factory = _runtime_factories[device.type]
    except KeyError:
        raise ValueError(f"no execution runtime for {device.type}") from None
    return factory(device)
