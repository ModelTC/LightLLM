import torch
from abc import ABC, abstractmethod


class BaseVitAttBackend(ABC):
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

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def vit_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        raise NotImplementedError("not impl")
