"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class BoundedAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
    """单种 bounded-attention 方案的数据操作，索引和槽位由调用方分配。"""

    def validate_bounded_att_commit(self, acceptance):

        pass
