"""Kimi Delta Attention (KDA) 线性注意力：一个因果深度卷积，后跟一个带有每通道门的 delta 规则（GLM-5-Next 的线性层）。

与 GDN 类似，KDA 保持每个请求的递归状态（卷积和 SSM），该状态在就地实现更新；调用者拥有它并提供每个层的视图，因此一个前向传播在每一步和每一层中恰好运行一次。
"""

from dataclasses import dataclass
from typing import Optional

import torch

PREFILL_KERNELS = ("triton", "flashkda", "auto")


@dataclass(frozen=True)
class KDAAttConfig:
    """KDA 层的静态配置，按 rank 计算。

    q、k、v 各有 num_heads 个 head_dim 维的注意力头，conv_kernel_size 是卷积宽度。
    衰减门以 lower_bound 为下限。state_mtp_step 指状态布局预留的投机深度，
    原生 SSM 状态为每个请求保留 state_mtp_step + 1 个槽。replay=True 选择重放状态，
    每个请求只用一个 SSM 槽，decode 将历史追加到 KDAAttInputs.replay_state。
    ssm_dtype 是 SSM 状态的 dtype。

    prefill_kernel 决定 prefill 是否使用 FlashKDA。"triton" 总是走 Triton，
    "flashkda" 要求使用 FlashKDA，加载失败时报错。"auto" 在 SM90 上为使用 BF16
    SSM 状态的 BF16 层启用 FlashKDA，且本步长度不得超过已验证的上限。
    """

    num_heads: int
    head_dim: int
    conv_kernel_size: int
    lower_bound: float
    dtype: torch.dtype
    state_mtp_step: int = 0
    replay: bool = False
    ssm_dtype: torch.dtype = torch.float32
    prefill_kernel: str = "triton"

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0 or self.conv_kernel_size <= 0:
            raise ValueError("num_heads, head_dim and conv_kernel_size must be positive")
        if self.state_mtp_step < 0:
            raise ValueError("state_mtp_step must be >= 0")
        if self.prefill_kernel not in PREFILL_KERNELS:
            raise ValueError(f"prefill_kernel must be one of {PREFILL_KERNELS}, got {self.prefill_kernel!r}")


@dataclass(frozen=True)
class KDAAttInputs:
    """一个层的运算符，按标记（prefill）或行（解码）顺序排列。

    mixed_qkv 在卷积前为 [tokens, 3 * num_heads * head_dim]；raw_gate [tokens, num_heads * head_dim] 和 raw_beta
    [tokens, num_heads] 是原始衰减和 beta 输入。conv_weight 是合并的 q/k/v 卷积权重，A_log 和 dt_bias 是门参数。conv_states 和
    ssm_states 是就地更新的层状态视图。重放解码用逻辑层索引 logic_layer_index 在 replay_state 中
    定位存储层并追加历史。
    """

    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    conv_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: Optional[torch.Tensor] = None
    replay_state: Optional[object] = None
    logic_layer_index: Optional[int] = None
