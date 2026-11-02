from contextlib import nullcontext
from dataclasses import dataclass

import torch


class CompletedFence:
    def query(self):
        return True

    def synchronize(self):
        pass


@dataclass(frozen=True)
class DeviceRuntime:
    device: torch.device

    def __post_init__(self):
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"no execution runtime for {self.device.type}")

    def allocate(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def stream(self, stream=None):
        return nullcontext() if self.device.type == "cpu" else torch.cuda.stream(stream)

    def record_completion(self):
        if self.device.type == "cpu":
            return CompletedFence()
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def wait(self, fence):
        if self.device.type == "cpu" or isinstance(fence, CompletedFence):
            fence.synchronize()
        else:
            torch.cuda.current_stream(self.device).wait_event(fence)
