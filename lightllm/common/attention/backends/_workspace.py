import torch
from lightllm.platform import get_device_runtime


class AttentionWorkspace:
    def __init__(self, device=None, *, runtime=None):
        self.runtime = runtime if runtime is not None else get_device_runtime(device)
        self.device = self.runtime.device
        self._buffers = {}
        self.objects = {}
        self._completion = None
        self._last_stream = None
        self._frozen = False
        self._pool = None

    def fork(self):
        """捕获实例借用当前池的地址；池后续扩容时，旧图仍持有原 tensor。

        同一槽位的借用者共享完成事件，因此共享 scratch 不允许并发执行；不同
        microbatch 槽位由模型创建不同的池。只有实际扩容才需要额外存储。
        """
        workspace = AttentionWorkspace(runtime=self.runtime)
        workspace._pool = self._pool or self
        workspace.objects = self.objects
        return workspace

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
        for name, size, dtype in resources.workspace:
            self.get(name, size, dtype)

    def synchronize(self):
        if self._pool is not None:
            return self._pool.synchronize()
        if self._completion is not None:
            self._completion.synchronize()

    def get(self, name: str, size: int, dtype=torch.uint8):
        if not name or size <= 0:
            raise ValueError("workspace name and positive size are required")
        key = name, dtype
        tensor = self._buffers.get(key)
        if tensor is None or tensor.numel() < size:
            if self._frozen:
                raise RuntimeError(f"captured workspace {name} cannot grow; reserve capacity before capture")
            tensor = (
                self._pool.get(name, size, dtype)
                if self._pool is not None
                else self.runtime.allocate(size, dtype=dtype)
            )
            self._buffers[key] = tensor

        return tensor
