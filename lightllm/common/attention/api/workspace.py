from contextlib import contextmanager
from typing import Callable, TypeVar, cast

import torch

from lightllm.platform import get_device_runtime

T = TypeVar("T")


class AttentionWorkspace:
    def __init__(self, device=None, *, runtime=None):
        self.runtime = runtime if runtime is not None else get_device_runtime(device)
        self.device = self.runtime.device
        self._buffers = {}
        self._objects = {}
        self._completion = None
        self._last_stream = None
        self._frozen = False
        self._pool = None

    def borrow_for_capture(self):
        """捕获实例借用当前池的地址；池后续扩容时，旧图仍持有原 tensor。

        同一槽位的借用者共享完成事件，因此共享 scratch 不允许并发执行；不同
        microbatch 槽位由模型创建不同的池。只有实际扩容才需要额外存储。
        """
        workspace = AttentionWorkspace(runtime=self.runtime)
        workspace._pool = self._pool or self
        workspace._objects = self._objects
        return workspace

    def get_object(self, name: str, factory: Callable[[], T], *, expected_type: type[T] | None = None) -> T:
        """Share objects across mutually exclusive contexts, checking their declared type.

        Forks share this live object pool, not a snapshot. Freezing a fork prevents that
        fork from adding objects or growing buffers; other buckets may still warm up.
        Existing objects must keep captured addresses stable even when the pool grows.
        """
        if expected_type is None:
            if not isinstance(factory, type):
                raise TypeError("workspace object factories require an expected_type")
            expected_type = factory
        if name not in self._objects:
            if self._frozen:
                raise RuntimeError(f"captured workspace object {name} must be created before capture")
            value = factory()
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"workspace object {name} requires {expected_type.__name__}, got {type(value).__name__}"
                )
            self._objects[name] = value
        value = self._objects[name]
        if not isinstance(value, expected_type):
            raise TypeError(f"workspace object {name} requires {expected_type.__name__}, got {type(value).__name__}")
        return cast(T, value)

    @contextmanager
    def lease(self, dependencies=()):
        self.begin()
        try:
            yield self
        finally:
            self.finish(dependencies)

    def begin(self):
        if self._pool is not None:
            return self._pool.begin()
        if self._completion is not None:
            stream = self.runtime.current_stream()
            if stream != self._last_stream:
                stream.wait_event(self._completion)

    def finish(self, dependencies=()):
        if self._pool is not None:
            return self._pool.finish(dependencies)
        stream = self.runtime.current_stream()
        if stream is not None:
            for event in dependencies:
                stream.wait_event(event)
            if self._completion is None:
                self._completion = self.runtime.create_event()
            self._last_stream = stream
            self._completion.record(self._last_stream)

    def freeze(self):
        """捕获后禁止替换任何地址；互斥 bucket 可继续借用已预留的大缓冲区。"""
        self._frozen = True

    def reserve(self, resources):
        for requirement in resources.workspace:
            self.scratch(requirement.name, requirement.elements, requirement.dtype)

    def synchronize(self):
        if self._pool is not None:
            return self._pool.synchronize()
        if self._completion is not None:
            self._completion.synchronize()

    def scratch(self, name: str, size: int, dtype=torch.uint8):
        if not name or size <= 0:
            raise ValueError("workspace name and positive size are required")
        key = name, dtype
        tensor = self._buffers.get(key)
        if tensor is None or tensor.numel() < size:
            if self._frozen:
                raise RuntimeError(f"captured workspace {name} cannot grow; reserve capacity before capture")
            tensor = (
                self._pool.scratch(name, size, dtype)
                if self._pool is not None
                else self.runtime.allocate(size, dtype=dtype)
            )
            self._buffers[key] = tensor

        return tensor
