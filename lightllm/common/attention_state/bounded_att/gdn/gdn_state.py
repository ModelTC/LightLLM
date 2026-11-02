from ..conv_ssm_state import ConvSSMState
from ...utils import dtype_name
from .gdn_pages_pool import GDNPagesPool


class GDNState(ConvSSMState):
    """Gated DeltaNet 的 Conv 历史、SSM 和 MTP 游标，支持不同的 K/V head 数及维度。"""

    page_pool_class = GDNPagesPool

    def get_state_type(self) -> str:
        """返回 GDN 算法及 Conv/SSM 存储精度。"""
        return f"gdn_conv_{dtype_name(dtype=self.config.conv_dtype)}" f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
