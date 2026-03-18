import torch
from contextlib import contextmanager
from enum import Enum
from lightllm.utils.log_utils import init_logger

try:
    from torch_memory_saver import (
        torch_memory_saver,
        configure_subprocess,
    )

    HAS_TORCH_MEMORY_SAVER = True

except ImportError:
    HAS_TORCH_MEMORY_SAVER = False
    pass

logger = init_logger(__name__)


class MemoryTag(Enum):
    KV_CACHE = "kv_cache"
    WEIGHT = "weights"
    GRAPH = "graph"

    def is_kv_cache(self):
        return self == MemoryTag.KV_CACHE

    def is_weight(self):
        return self == MemoryTag.WEIGHT

    def is_graph(self):
        return self == MemoryTag.GRAPH

    def __str__(self):
        return self.value


class TorchMemorySaverWrapper:
    def __new__(cls, enable_torch_memory_saver: bool = False):
        if enable_torch_memory_saver:
            assert (
                HAS_TORCH_MEMORY_SAVER
            ), "torch_memory_saver is not installed, please install it via `pip install torch_memory_saver`."
            return _TorchMemorySaver()
        else:
            return _TorchMemorySaverFake()


class _TorchMemorySaver:
    def configure_subprocess(self):
        return configure_subprocess()

    def region(self, tag: MemoryTag, enable_cpu_backup: bool = False):
        return torch_memory_saver.region(tag=tag.value, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, graph_obj: torch.cuda.CUDAGraph, **kwargs):
        return torch_memory_saver.cuda_graph(cuda_graph=graph_obj, **kwargs, tag=MemoryTag.GRAPH.value)

    def disable(self):
        return torch_memory_saver.disable()

    def pause(self, tag: MemoryTag):
        return torch_memory_saver.pause(tag=tag.value)

    def resume(self, tag: MemoryTag):
        return torch_memory_saver.resume(tag=tag.value)


class _TorchMemorySaverFake:
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: MemoryTag, enable_cpu_backup: bool = False):
        yield

    def cuda_graph(self, graph_obj: torch.cuda.CUDAGraph, **kwargs):
        return torch.cuda.graph(graph_obj, **kwargs)

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: MemoryTag):
        logger.warning("torch_memory_saver is not enabled, pause is not supported.")
        return

    def resume(self, tag: MemoryTag):
        logger.warning("torch_memory_saver is not enabled, resume is not supported.")
        return
