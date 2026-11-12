from abc import ABC, abstractmethod

import torch

from .inputs import VarlenAttentionInputs


class VarlenAttentionOp(ABC):
    """无持久 KV 的双向 dense attention，供视觉和音频编码器使用。

    Q/K/V 与变长边界由调用方直接传入，不需要文本推理的请求表或 cache context。
    实现不保留批次 tensor；调用方持有 out，返回值必须引用同一块输出存储。
    SDPA / xFormers 需要在 CPU 读取序列边界；只有声明 supports_graph 的实现可整段捕获。
    """

    supports_graph = False

    @classmethod
    @abstractmethod
    def check_environment(cls, caps) -> None:
        """检查运行环境；不支持时抛出 UnsupportedAttention。"""
        pass

    @classmethod
    @abstractmethod
    def check_support(cls, inputs: VarlenAttentionInputs, caps=None) -> None:
        """初始化或探针校验输入；不支持时抛出 UnsupportedAttention。"""
        pass

    @abstractmethod
    def forward(self, inputs: VarlenAttentionInputs, *, out: torch.Tensor) -> torch.Tensor:
        pass
