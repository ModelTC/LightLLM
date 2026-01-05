from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Optional


class BaseAttBackend:
    """
    用于创建支持各种不同的AttBackend, 如 fa3, flashinfer, triton 实现等，
    这个是单列模式, 每种backend只有一个实例
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        """
        重写__new__方法实现单例模式
        """
        # 检查是否已经有该类的实例
        if cls not in cls._instances:
            # 创建新实例并存储
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        # 返回已有的实例
        return cls._instances[cls]

    def create_att_prefill_state(self) -> "BasePrefillAttState":
        raise NotImplementedError("not impl")

    def create_att_decode_state(self) -> "BaseDecodeAttState":
        raise NotImplementedError("not impl")


@dataclass
class BasePrefillAttState(ABC):
    backend: BaseAttBackend = None
    infer_state = None

    @abstractmethod
    def init_state(self):
        pass

    @abstractmethod
    def copy_for_prefill_cuda_graph(self, new_state: "BasePrefillAttState"):
        pass

    @abstractmethod
    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.tensor,
        v: torch.tensor,
        layer_weight,
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
        use_alibi=False,
    ) -> torch.Tensor:
        raise NotImplementedError("not impl")


@dataclass
class BaseDecodeAttState(ABC):
    backend: BaseAttBackend = None
    infer_state = None

    @abstractmethod
    def init_state(self):
        pass

    @abstractmethod
    def copy_for_decode_cuda_graph(self, new_state: "BaseDecodeAttState"):
        pass

    @abstractmethod
    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        out: Optional[torch.Tensor] = None,
        alloc_func=torch.empty,
        use_alibi=False,
    ) -> torch.Tensor:
        pass
