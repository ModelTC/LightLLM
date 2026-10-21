import torch
from contextlib import contextmanager
from enum import Enum
from lightllm.utils.log_utils import init_logger

try:
    from torch_memory_saver import (
        torch_memory_saver,
        configure_subprocess as tms_configure_subprocess,
    )

    HAS_TORCH_MEMORY_SAVER = True

except ImportError:
    HAS_TORCH_MEMORY_SAVER = False
    pass

logger = init_logger(__name__)


class MemoryTag(Enum):
    # torch_memory_saver 通过 tag 区分不同类型的显存区域，后续 pause/resume
    # 可以只针对某一类内存做释放和恢复。
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
    # 统一返回真实实现或空实现，调用方不需要到处判断
    # enable_torch_memory_saver 是否开启。
    def __new__(cls, enable_torch_memory_saver: bool = False):
        if enable_torch_memory_saver:
            assert (
                HAS_TORCH_MEMORY_SAVER
            ), "torch_memory_saver is not installed, please install it via `pip install torch_memory_saver`."
            return _TorchMemorySaver()
        else:
            return _TorchMemorySaverFake()


class _TorchMemorySaver:
    @contextmanager
    def configure_subprocess(self):
        # 子进程启动需要放在该上下文里，让 torch_memory_saver 在 worker
        # 进程中完成必要的初始化。
        with tms_configure_subprocess():
            yield

    def region(self, tag: MemoryTag, enable_cpu_backup: bool = False):
        # 记录这个上下文内产生的显存分配；enable_cpu_backup 用于需要
        # pause 后还能恢复内容的区域，比如权重。
        return torch_memory_saver.region(tag=tag.value, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, graph_obj: torch.cuda.CUDAGraph, **kwargs):
        # CUDA graph 的 private pool 也单独打 tag，避免和普通权重/KV cache
        # 的显存管理混在一起。
        return torch_memory_saver.cuda_graph(cuda_graph=graph_obj, **kwargs, tag=MemoryTag.GRAPH.value)

    def disable(self):
        return torch_memory_saver.disable()

    def pause(self, tag: MemoryTag):
        return torch_memory_saver.pause(tag=tag.value)

    def resume(self, tag: MemoryTag):
        return torch_memory_saver.resume(tag=tag.value)


class _TorchMemorySaverFake:
    # 未开启 torch_memory_saver 时保持相同接口，保证调用方逻辑完全一致。
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
