from typing import Callable, TypeVar, cast

import torch

from lightllm.platform import get_device_runtime
from lightllm.common.execution import ExecutionScope

T = TypeVar("T")


class AttentionWorkspace:
    def __init__(self, device=None, *, runtime=None, execution=None):
        self.runtime = runtime if runtime is not None else get_device_runtime(device)
        self.execution = execution if execution is not None else ExecutionScope(self.runtime)
        self.device = self.runtime.device
        self._buffers = {}
        self._objects = {}
        self._frozen = False
        self._pool = None

    def borrow_for_capture(self):
        """捕获实例借用当前池的地址；池后续扩容时，旧图仍持有原 tensor。

        同一槽位的借用者共享完成事件，因此共享 scratch 不允许并发执行；不同
        microbatch 槽位由模型创建不同的池。只有实际扩容才需要额外存储。
        """
        workspace = AttentionWorkspace(runtime=self.runtime, execution=self.execution)
        workspace._pool = self._pool or self
        return workspace

    def get_object(self, name: str, factory: Callable[[], T], *, expected_type: type[T] | None = None) -> T:
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

    def freeze(self):
        """捕获后禁止替换任何地址；互斥 bucket 可继续借用已预留的大缓冲区。"""
        self._frozen = True

    def clear_objects(self):
        if self._frozen:
            raise RuntimeError("cannot clear objects owned by a captured workspace")
        self._objects.clear()

    def reserve(self, resources):
        for requirement in resources.workspace:
            self.scratch(requirement.name, requirement.elements, requirement.dtype)

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
