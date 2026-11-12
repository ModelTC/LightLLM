"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

from dataclasses import fields

from lightllm.common.attention.api import CapabilityResult
from lightllm.platform import Platform

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class BoundedAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
    """单种 bounded-attention 方案的数据操作，索引和槽位由调用方分配。"""

    def capture_snapshot_views(self, req_indices):
        raise NotImplementedError(f"{type(self).__name__} does not implement capture snapshots")

    def describe_layout(self):
        """显式布局标识与配置，不用 Python 类名判断物理兼容性。"""
        return self.storage_layout + (
            tuple(
                (field.name, getattr(self.config, field.name))
                for field in fields(self.config)
                if field.name not in ("logical_layer_ids", "logical_layer_groups")
            ),
        )

    def check_execution(self, caps, execution):
        if caps.platform is not Platform.CUDA:
            return CapabilityResult.reject(f"{self.storage_layout[0]} state operations require CUDA")
        return CapabilityResult.ok()
