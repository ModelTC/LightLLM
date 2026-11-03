from dataclasses import dataclass
from math import prod

import torch

from lightllm.common.device_runtime import get_device_runtime


@dataclass(frozen=True)
class TensorStorageSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    fill: int | float | None = None
    source: torch.Tensor | None = None
    fill_regions: tuple = ()

    @property
    def nbytes(self):
        return prod(self.shape) * self.dtype.itemsize


def allocate_buffers(specs, device):
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("storage field names must be unique")
    runtime = get_device_runtime(device)
    buffers = {}
    for spec in specs:
        if spec.source is None:
            buffers[spec.name] = runtime.allocate(spec.shape, spec.dtype, spec.fill)
        else:
            if spec.source.shape != spec.shape or spec.source.dtype != spec.dtype:
                raise ValueError(f"initializer does not match storage field {spec.name}")
            buffers[spec.name] = spec.source if runtime.device.type == "meta" else runtime.copy_to(spec.source)
        for index, value in spec.fill_regions:
            buffers[spec.name][index].fill_(value)
    return buffers


def allocate_state_storage(state, token_capacity):
    buffers = allocate_buffers(state.get_storage_specs(token_capacity), state.device)
    state.bind_storage(buffers, token_capacity)


def release_state_storage(state):
    if not state.storage_fields:
        raise NotImplementedError(f"{type(state).__name__} does not declare allocation ownership")
    for name in state.storage_fields:
        setattr(state, name, None)
    state.capacity = 0


def allocate_runtime_storage(state, device):
    specs = tuple(spec for spec in state.get_runtime_storage_specs() if getattr(state, spec.name).is_meta)
    if specs:
        state.bind_runtime_storage(allocate_buffers(specs, device))
    return state


class StatePagesPool:
    def __init__(self, specs, size, device="cpu", pin_memory=True):
        from .bounded_att.utils import create_page_pool_tensor

        if size < 0:
            raise ValueError("pool size must be nonnegative")
        self.size = size
        self.fields = tuple(spec.name for spec in specs)
        for spec in specs:
            setattr(
                self,
                spec.name,
                create_page_pool_tensor((size, *spec.shape), spec.dtype, device, pin_memory),
            )

    def register_pin_memory(self):
        from .bounded_att.utils import register_page_pool_tensor

        for name in self.fields:
            register_page_pool_tensor(getattr(self, name))


def initialize_state_page_pools(state, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
    assert state.big_pages_pool is None and state.small_pages_pool is None, "page pools already initialized"
    specs = state.get_snapshot_storage_specs()
    big = StatePagesPool(specs, big_page_capacity, device, pin_memory) if specs and big_page_capacity else None
    small = StatePagesPool(specs, small_page_capacity, device, pin_memory) if specs and small_page_capacity else None
    state.big_pages_pool, state.small_pages_pool = big, small


def release_state_page_pools(state):
    state.big_pages_pool = state.small_pages_pool = None
