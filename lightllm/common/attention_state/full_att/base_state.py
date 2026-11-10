from __future__ import annotations

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class FullAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
    """单种 full-attention 存储方案的数据接口，不分配 token 索引或维护请求映射。

    所有表示 token 索引的参数都须为调用方准备的同设备 Tensor，位于对应存储设备上；
    State 不转换设备、dtype 或连续布局，int32/int64 由具体算子支持，DP 目标索引须为 int64。
    传输操作的设备限制由具体实现声明，CPU 页位于主机内存不改变索引与存储同设备的契约。
    """

    def describe_layout(self):
        """显式布局标识与配置，不用 Python 类名判断物理兼容性。"""
        from dataclasses import fields

        return self.storage_layout + (
            tuple(
                (field.name, getattr(self.config, field.name))
                for field in fields(self.config)
                if field.name not in ("logical_layer_ids", "logical_layer_groups")
            ),
        )

    def check_execution(self, caps, execution):
        from lightllm.platform import Platform
        from lightllm.common.attention.api import CapabilityResult

        if caps.platform is not Platform.CUDA:
            return CapabilityResult.reject(f"{self.storage_layout[0]} state operations require CUDA")
        return CapabilityResult.ok()
