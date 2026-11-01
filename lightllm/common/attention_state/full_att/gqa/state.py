from ..state import KVWritableState, TokenAttentionState


class GQAStorageState(KVWritableState, TokenAttentionState):
    """兼容现有 K/V 属性，底层引用属于 TokenStorage，计算与传输按相同字段规格取视图。"""

    @property
    def storage(self):
        return self.resource

    @property
    def k_storage(self):
        return self.storage.buffers.get("k")

    @k_storage.setter
    def k_storage(self, buffer):
        self.storage.buffers["k"] = buffer

    @property
    def v_storage(self):
        return self.storage.buffers.get("v")

    @v_storage.setter
    def v_storage(self, buffer):
        self.storage.buffers["v"] = buffer
