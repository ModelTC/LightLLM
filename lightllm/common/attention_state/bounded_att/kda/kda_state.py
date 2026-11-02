from ..conv_ssm_state import ConvSSMState
from ...utils import dtype_name
from .kda_pages_pool import KDAPagesPool


class KDAState(ConvSSMState):
    """KDA 的 Conv 历史、方形 SSM 和 MTP 游标，要求 Q/K/V head 数及维度相同。"""

    page_pool_class = KDAPagesPool

    def get_state_type(self) -> str:
        """返回 KDA 算法及 Conv/SSM 存储精度。"""
        return f"kda_conv_{dtype_name(dtype=self.config.conv_dtype)}" f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
