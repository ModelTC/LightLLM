from __future__ import annotations

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class FullAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
    """单种 full-attention 存储方案的数据接口，不分配 token 索引或维护请求映射。

    搬运索引与原生存储使用同一执行域：CPU 存储使用 CPU 索引，CUDA 存储使用 CUDA 索引。
    State 不转换索引的设备、dtype 或连续布局；原生拷贝支持 int32/int64。
    CUDA 向主机页搬运时，外部页必须 pinned 或注册。
    """

    resource_kind = "kv"
    storage_fields = ()

    def write_full_att_kv(self, logical_layer_id, token_indices, kv):

        self.bind_kv_writer(logical_layer_id)(token_indices=token_indices, kv=kv)

    def release_storage(self):

        if not self.storage_fields:
            raise NotImplementedError(f"{type(self).__name__} does not declare allocation ownership")
        for name in self.storage_fields:
            setattr(self, name, None)
        self.capacity = 0

    def get_transfer_fields(self):

        return None

    def get_attention_format(self):

        return None

    def get_token_copy_fields(self):
        return self.get_transfer_fields()

    def get_fixed_storage_bytes(self):
        return 0

    def _copy_pages(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None, pad_tail=False):
        from ..token_copy import TokenPageCopyPlan, TransferBatch

        fields = self.get_transfer_fields()
        if fields is None:
            raise NotImplementedError("this state requires a dedicated copy provider")
        batch = TransferBatch.prepare(token_indices, page_indices, page_readies, page_count=page_buffers.shape[0])
        plan = TokenPageCopyPlan.bind(fields, layout, page_buffers, write, pad_tail=pad_tail)
        plan.run(batch)

    def write_to_cpu(self, token_indices, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None):
        if cpu_page_ready_flags is not None:
            if len(cpu_page_ready_flags) != len(cpu_page_indices):
                raise ValueError("page and ready counts differ")
            if all(cpu_page_ready_flags):
                return
        self._copy_pages(token_indices, cpu_page_buffers, cpu_page_indices, layout, True, cpu_page_ready_flags)

    def read_from_cpu(self, cpu_page_buffers, cpu_page_indices, token_indices, layout):
        self._copy_pages(token_indices, cpu_page_buffers, cpu_page_indices, layout, False)

    def write_to_pd_page(self, token_indices, pd_page_buffers, pd_page_index, layout):
        self._copy_pages(token_indices, pd_page_buffers, [pd_page_index], layout, True, pad_tail=True)

    def read_from_pd_page(self, pd_page_buffers, token_indices, pd_page_index, layout):
        self._copy_pages(token_indices, pd_page_buffers, [pd_page_index], layout, False, pad_tail=True)

    def copy_full_att_tokens(self, src_indices, dst_indices):
        self.copy_tokens_from_dp(src_indices, dst_indices, self)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):

        from ..token_copy import TokenPageCopyPlan, TransferBatch

        if remote_dp_full_att_state is not self:
            if (
                self.config != remote_dp_full_att_state.config
                or self.get_attention_format() != remote_dp_full_att_state.get_attention_format()
            ):
                raise ValueError("direct State copies require equivalent representations, shards and layer groups")
        plan = TokenPageCopyPlan.for_tokens(
            remote_dp_full_att_state.get_token_copy_fields(), self.get_token_copy_fields()
        )
        plan.run(TransferBatch(src_token_indices, dst_token_indices))
