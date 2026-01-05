import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


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
class AttControl:
    """
    prefill_att 和 decode_att 的入参，用于控制att backend 内部的行为, 选择正确的att 实现。
    """

    use_alibi: bool = (False,)


@dataclass
class BasePrefillAttState(ABC):

    backend: BaseAttBackend = None
    infer_state: "InferStateInfo" = None

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
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        raise NotImplementedError("not impl")


@dataclass
class BaseDecodeAttState(ABC):
    backend: BaseAttBackend = None
    infer_state: "InferStateInfo" = None

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
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        pass


@dataclass
class AttControl:
    """
    prefill_att 和 decode_att 的入参，用于控制att backend 内部的行为, 选择正确的att 实现。
    """

    use_alibi: bool = False
