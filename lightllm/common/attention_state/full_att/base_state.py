from __future__ import annotations

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class FullAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
    """单种 full-attention 存储方案的数据接口，不分配 token 索引或维护请求映射。

    所有表示 token 索引的参数都须为调用方准备的 CUDA Tensor，位于对应存储设备上；
    State 不转换设备、dtype 或连续布局，int32/int64 由具体算子支持，DP 目标索引须为 int64。
    CPU 页位于主机内存不改变 token 索引的 CUDA 契约。
    """
