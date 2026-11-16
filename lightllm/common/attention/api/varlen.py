from abc import ABC, abstractmethod

import torch

from .inputs import VarlenAttentionInputs


class VarlenAttentionOp(ABC):
    """无持久 KV 的双向 dense attention，供视觉和音频编码器使用。

    Q/K/V 与变长边界由调用方直接传入，不需要文本推理的请求表或 cache context。
    实现不保留批次 tensor；调用方持有 out，返回值必须引用同一块输出存储。
    out 的 shape/dtype/device 与 query 一致，head_dim 连续，其他维度允许无重叠的 stride；
    不允许与 Q/K/V 共享存储。每次 forward 在启动 kernel 前验证这些 metadata。
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
        """检查 dtype/head dim/platform 等能力限制；不读取或校验序列边界值。"""
        pass

    @abstractmethod
    def forward(self, inputs: VarlenAttentionInputs, *, out: torch.Tensor) -> torch.Tensor:
        """每次执行校验当前输入布局，然后写入 out；Graph replay 也须重新执行设备值检查。"""
        pass


def validate_varlen_output(inputs: VarlenAttentionInputs, out: torch.Tensor) -> None:
    """Validate metadata only; out may have padding between heads/tokens, but cannot alias inputs."""
    query = inputs.query
    if out.shape != query.shape or out.dtype != query.dtype or out.device != query.device:
        raise ValueError("varlen out must match query shape, dtype and device")
    if out.layout != torch.strided or out.stride(-1) != 1:
        raise ValueError("varlen out must have a contiguous head dimension")
    extent = 1
    for stride, size in sorted(zip(out.stride(), out.shape)):
        if size > 1:
            if stride < extent:
                raise ValueError("varlen out must have non-overlapping strides")
            extent = stride * (size - 1) + extent
    if out.numel():
        storage = out.untyped_storage().data_ptr()
        if any(storage == tensor.untyped_storage().data_ptr() for tensor in (inputs.query, inputs.key, inputs.value)):
            raise ValueError("varlen out must not share storage with query, key or value")
