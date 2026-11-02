import torch

from ..conv_ssm_state import ConvSSMState
from ..conv_ssm_types import ConvSSMStateConfig
from ...utils import dtype_name


class KDAState(ConvSSMState):
    """KDA 的 Conv 历史、方形 SSM 和 MTP 游标，要求 Q/K/V head 数及维度相同。"""

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        assert (
            config.num_k_heads == config.num_v_heads and config.key_dim == config.value_dim
        ), "KDA requires equal Q/K/V heads and dimensions"
        super().__init__(config=config, request_capacity=request_capacity, mtp_step=mtp_step, device=device)

    def get_state_type(self) -> str:
        """返回 KDA 算法及 Conv/SSM 存储精度。"""
        return f"kda_conv_{dtype_name(dtype=self.config.conv_dtype)}" f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
