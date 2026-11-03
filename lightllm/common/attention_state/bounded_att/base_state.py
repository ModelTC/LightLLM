"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations


from .interface import CoreInterface, CpuCacheInterface, PdInterface


class BoundedAttentionState(CoreInterface, CpuCacheInterface, PdInterface):
    """单种 bounded-attention 方案的数据操作，索引和槽位由调用方分配。"""

    runtime_storage_fields = ()

    def get_unallocated_runtime_bytes(self):
        if not self.runtime_storage_fields:
            raise NotImplementedError("bounded state must declare runtime allocation ownership")
        return sum(
            value.numel() * value.element_size()
            for name in self.runtime_storage_fields
            if (value := getattr(self, name)) is not None and value.is_meta
        )

    def bind_runtime_storage(self, buffers):
        self.__dict__.update(buffers)

    def get_runtime_storage_specs(self):
        raise NotImplementedError("bounded state must declare native runtime storage")

    @property
    def resource_kind(self):

        raise NotImplementedError("bounded state must declare its logical resource kind")

    def get_runtime_copy_fields(self, request):
        return self.get_runtime_transfer_view().at(request)

    def get_snapshot_fields(self, page_kind, page):
        return self.get_snapshot_transfer_view(page_kind).at(page)

    def get_runtime_transfer_view(self):
        raise NotImplementedError("bounded state must declare native runtime fields")

    def get_snapshot_transfer_view(self, page_kind):
        raise NotImplementedError("bounded state must declare native snapshot fields")

    def get_checkpoint_bytes(self):
        if not self.get_cpu_transfer_regions():
            return 0
        raise NotImplementedError("state must declare the native size of one checkpoint")
