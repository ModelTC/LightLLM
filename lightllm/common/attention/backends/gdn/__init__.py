"""Gated DeltaNet (GDN) 线性注意力：一个因果深度卷积后跟门控 delta 规则。

与 softmax 注意力不同，GDN 为每个请求保留了递归状态而不是 KV 缓存：一个卷积状态
保存最近的卷积输入，以及每个值头的 SSM 状态矩阵。一个实现会就地读取和更新该状态；
调用者拥有它并为每一层提供视图。因此，前向传播必须在每一步和每一层中恰好运行一次。
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GDNAttConfig:
    """按 rank配置 GDN 层的静态配置。

    查询和键具有 num_k_heads 个 head_k_dim 头；值具有 num_v_heads（num_k_heads 的倍数）个 head_v_dim 头。conv_kernel_dim 是卷积宽度，
    activation 是其激活函数。
    state_mtp_step 是状态布局预留的投机深度：卷积状态保留那么多额外的尾列，
    而原生 SSM 状态为每个请求保留 state_mtp_step + 1 个槽。replay=True
    选择重放状态：每个请求一个 SSM 槽，解码将历史追加到
    GDNAttInputs.replay_state 而不是更新 SSM 状态。
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_dim: int
    activation: str
    dtype: torch.dtype
    state_mtp_step: int = 0
    replay: bool = False

    def __post_init__(self) -> None:
        if (
            self.num_k_heads <= 0
            or self.num_v_heads <= 0
            or self.head_k_dim <= 0
            or self.head_v_dim <= 0
            or self.conv_kernel_dim <= 0
        ):
            raise ValueError("num_k_heads, num_v_heads, head_k_dim, head_v_dim and conv_kernel_dim must be positive")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.state_mtp_step < 0:
            raise ValueError("state_mtp_step must be >= 0")


@dataclass(frozen=True)
class GDNAttInputs:
    """一层的操作数，在标记（prefill）或行（解码）顺序中。

    mixed_qkv 是卷积前的 [tokens, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim]；
    a 和 b 是原始门控输入 [tokens, num_v_heads] 和 z 是输出门 [tokens, num_v_heads,
    head_v_dim]，解码内核可以读取。conv_weight/conv_bias, A_log 和 dt_bias 是
    层的参数。conv_states 和 ssm_states 是层的状态视图，按请求槽索引，
    前向传播会就地更新。重放解码用逻辑层索引 logic_layer_index 在 replay_state 中
    定位存储层，读取并追加历史。
    """

    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    z: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: Optional[torch.Tensor]
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    conv_states: torch.Tensor
    ssm_states: Optional[torch.Tensor] = None
    replay_state: Optional[object] = None
    logic_layer_index: Optional[int] = None
