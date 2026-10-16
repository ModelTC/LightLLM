"""PyTorch MPS runtime on Apple's single default Metal command queue.

Stream scopes share the default queue; events mark GPU progress without
forcing a device-wide CPU wait at every record.
CUDA graph capture and IPC are deliberately unavailable on this runtime.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import time
import threading
import torch

from .transfers import HostTransferQueue


class MetalEvent:
    def __init__(self, **kwargs):
        self._recorded = False
        self._enable_timing = kwargs.get("enable_timing", False)
        self._time = None
        self._event = torch.mps.Event(enable_timing=False)

    def record(self, stream=None):
        if self._enable_timing:
            # Synchronized host timestamps preserve the previous timing
            # contract on macOS 14, where native elapsed_time can block.
            # Ordinary dependency events never take this device-wide barrier.
            torch.mps.synchronize()
            self._time = time.perf_counter()
        self._event.record()
        self._recorded = True

    def synchronize(self):
        self._event.synchronize()

    def query(self):
        return self._event.query()

    def wait(self, stream=None):
        self._event.wait()

    def elapsed_time(self, end_event):
        if self._time is None or end_event._time is None:
            raise RuntimeError("Record both events with enable_timing=True before measuring elapsed time")
        end_event.synchronize()
        return (end_event._time - self._time) * 1000


class MetalStream:
    def __init__(self, synchronize=None):
        self._synchronize = synchronize or torch.mps.synchronize

    def synchronize(self):
        self._synchronize()

    def wait_event(self, event):
        event.wait(self)

    def wait_stream(self, stream):
        # Every MetalStream is the same underlying MPS queue. Submission
        # order already supplies this dependency; a CPU barrier is unnecessary.
        pass

    def record_event(self, event=None):
        event = event or MetalEvent()
        event.record(self)
        return event


class MetalRuntime:
    name = "metal"
    device_type = "mps"
    device = "mps"
    distributed_backend = "gloo"
    supports_graphs = False
    supports_tensor_cache = False
    supports_pinned_memory = False
    event_type = MetalEvent
    stream_type = MetalStream
    # Used in annotations only. Capture methods fail explicitly.
    graph_type = object
    graph_context_type = object

    def __init__(self):
        self._host_transfers = HostTransferQueue()
        self._stream = MetalStream(self.synchronize)
        self._submission_state = threading.local()
        self.layer_submission_hook = self._submit_decode_layer

    def _submit_decode_layer(self, layer_index, batch_size):
        # Submit early enough to overlap Python encoding with GPU execution.
        # Larger batches amortize command-buffer/event overhead over four
        # layers. Public non-timing Event.record commits without a CPU wait.
        interval = 1 if batch_size <= 4 else 4
        if (layer_index + 1) % interval:
            return
        event = getattr(self._submission_state, "event", None)
        if event is None:
            event = torch.mps.Event(enable_timing=False)
            self._submission_state.event = event
        event.record()

    def device_at(self, index):
        if int(index) != 0:
            raise ValueError("Metal supports one device (index 0)")
        return self.device

    def move(self, value, device=None, non_blocking=False, memory_format=torch.preserve_format):
        if device is not None and device not in (0, "mps", "mps:0", torch.device("mps")):
            raise ValueError(f"Unsupported Metal device: {device}")
        if non_blocking:
            result = self._host_transfers.move(value, memory_format)
            if result is not None:
                return result
        return value.to(device=self.device, non_blocking=False, memory_format=memory_format)

    def is_device_tensor(self, value):
        return value.device.type == "mps"

    def generator(self, *args, **kwargs):
        # MPS has no per-device torch.Generator. Sampling uses the default MPS
        # generator seeded by manual_seed, without passing this CPU generator.
        kwargs.pop("device", None)
        return torch.Generator(*args, **kwargs)

    def exponential_(self, tensor, generator=None):
        if generator is None:
            return tensor.exponential_()
        noise = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        noise.exponential_(generator=generator)
        return tensor.copy_(noise)

    def is_available(self):
        return torch.backends.mps.is_available()

    def init(self):
        if not self.is_available():
            raise RuntimeError("Apple Metal requires a macOS arm64 PyTorch build with MPS available")

    lazy_init = init

    def is_initialized(self):
        return self.is_available()

    def current_device(self):
        return 0

    def set_device(self, index):
        self.device_at(index)
        self.init()

    def device_count(self):
        return int(self.is_available())

    def get_device_name(self, device=None):
        return "Apple Metal"

    def get_device_capability(self, device=None):
        # CUDA compute capabilities have no equivalent on Metal.
        return (0, 0)

    def get_device_properties(self, device=None):
        return SimpleNamespace(name=self.get_device_name(), total_memory=torch.mps.recommended_max_memory())

    def synchronize(self, device=None):
        self._host_transfers.synchronize()

    def current_stream(self, device=None):
        return self._stream

    def stream_scope(self, stream):
        return nullcontext(self._stream)

    def new_stream(self, **kwargs):
        return self._stream

    def new_event(self, **kwargs):
        return MetalEvent(**kwargs)

    def device_scope(self, device):
        self.device_at(device if isinstance(device, int) else 0)
        return nullcontext()

    def new_graph(self, *args, **kwargs):
        raise NotImplementedError("Graph capture is unavailable on Metal")

    capture_graph = new_graph
    graph_pool_handle = new_graph

    def is_current_stream_capturing(self):
        return False

    def empty_cache(self):
        self.synchronize()
        torch.mps.empty_cache()

    def memory_allocated(self, device=None):
        return torch.mps.current_allocated_memory()

    def memory_reserved(self, device=None):
        return torch.mps.driver_allocated_memory()

    def max_memory_reserved(self, device=None):
        raise NotImplementedError("MPS does not expose peak reserved memory statistics")

    def reset_peak_memory_stats(self, device=None):
        raise NotImplementedError("MPS does not expose peak allocator statistics")

    def mem_get_info(self, device=None):
        total = torch.mps.recommended_max_memory()
        return max(0, total - torch.mps.driver_allocated_memory()), total

    def ipc_collect(self):
        raise NotImplementedError("CUDA IPC is unavailable on Metal")

    def manual_seed(self, seed):
        torch.mps.manual_seed(seed)

    manual_seed_all = manual_seed

    def is_bf16_supported(self):
        import platform

        return tuple(map(int, platform.mac_ver()[0].split(".")[:1])) >= (14,)

    def range_start(self, name):
        return None

    def range_end(self, marker):
        return None


def _factory(name):
    def call(self, *args, **kwargs):
        device = kwargs.get("device")
        if isinstance(device, int):
            kwargs["device"] = self.device_at(device)
        # Pinning is a CUDA host transfer optimization; MPS uses unified memory.
        kwargs.pop("pin_memory", None)
        return getattr(torch, name)(*args, **kwargs)

    call.__name__ = name
    return call


for _name in (
    "empty zeros ones full arange tensor randn randint rand empty_like zeros_like "
    "ones_like full_like randn_like rand_like eye linspace logspace as_tensor"
).split():
    setattr(MetalRuntime, _name, _factory(_name))
