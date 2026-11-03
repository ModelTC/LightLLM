from __future__ import annotations

from .interface import CoreInterface, CpuCacheInterface, PdInterface


class FullAttentionState(CoreInterface, CpuCacheInterface, PdInterface):
    """单种 full-attention 存储方案的数据接口，不分配 token 索引或维护请求映射。

    搬运索引与原生存储使用同一执行域：CPU 存储使用 CPU 索引，CUDA 存储使用 CUDA 索引。
    State 不转换索引的设备、dtype 或连续布局；原生拷贝支持 int32/int64。
    CUDA 向主机页搬运时，外部页必须 pinned 或注册。
    """

    resource_kind = "kv"
    storage_fields = ()

    def bind_storage(self, buffers, token_capacity):
        self.__dict__.update(buffers)
        self.capacity = token_capacity

    def get_transfer_fields(self):
        raise NotImplementedError("full attention must declare its native transfer fields")

    def get_attention_format(self):

        return None

    def get_token_copy_fields(self):
        return self.get_transfer_fields()

    def get_fixed_storage_bytes(self, device=None):
        return 0
