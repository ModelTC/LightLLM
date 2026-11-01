from copy import copy
from functools import lru_cache

from ..utils import deepcopy_result
from .base_state import BoundedAttentionState
from .snapshot_transfer import SnapshotTransfer
from .storage import BoundedStorage


class ComposedBoundedState(BoundedAttentionState):
    """组合固定运行态、快照存储、算法语义和原始字段搬运。

    State 不分配请求/页索引，不维护前缀引用或发布可用缓存。
    所有异步复制完成前，调用方须保留源和目标槽位。
    """

    def __init__(self, config, spec, runtime, request_capacity, mtp_step, device, page_pool_factory):
        self.config = config
        self.state_name = None
        self.resource = BoundedStorage(
            spec=spec,
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=page_pool_factory,
        )
        self.runtime = runtime
        self.transfer = SnapshotTransfer()
        self.runtime.prepare(self.resource)

    def __copy__(self):
        """复制资源包装以便 IPC 重绑页池；Tensor 和 runtime 仍共享，这不是状态快照。"""
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.resource = copy(self.resource)
        return result

    @property
    def request_capacity(self):
        return self.resource.request_capacity

    @property
    def mtp_step(self):
        return self.resource.mtp_step

    @property
    def big_pages_pool(self):
        return self.resource.big_pages_pool

    @big_pages_pool.setter
    def big_pages_pool(self, pool):
        self.resource.big_pages_pool = pool

    @property
    def small_pages_pool(self):
        return self.resource.small_pages_pool

    @small_pages_pool.setter
    def small_pages_pool(self, pool):
        self.resource.small_pages_pool = pool

    def get_state_type(self):
        return self.resource.spec.encoding

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        self.resource.initialize_page_pools(
            big_page_capacity=big_page_capacity,
            small_page_capacity=small_page_capacity,
            device=device,
            pin_memory=pin_memory,
        )

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id in self.resource.layer_indices

    def get_layer_view(self, logical_layer_id):
        return self.runtime.layer_view(self.resource, logical_layer_id)

    def reset_bounded_att_state(self, req_indices):
        self.runtime.reset(self.resource, req_indices)

    def save_bounded_att_runtime_to_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        """将已执行的绝对 token 长度处的状态保存到预留槽位，不发布缓存条目。

        长度不包含尚未执行的采样 token；调用方须满足具体 runtime 的快照边界。
        """
        self.runtime.save(
            storage=self.resource,
            req_indices=req_indices,
            state_token_lens=state_token_lens,
            page_kind=page_kind,
            page_slot_ids=page_slot_ids,
        )

    def restore_bounded_att_runtime_from_pages(self, req_indices, state_token_lens, page_kind, page_slot_ids):
        """从快照恢复继续计算位置，原快照仍保留，不转移页池槽位所有权。"""
        self.runtime.restore(
            storage=self.resource,
            req_indices=req_indices,
            state_token_lens=state_token_lens,
            page_kind=page_kind,
            page_slot_ids=page_slot_ids,
        )

    def commit_bounded_att_state(self, req_indices, accepted_steps):
        """选择已接受的候选状态；不复制到基准槽位，也不将该位置发布为缓存。"""
        self.runtime.commit(self.resource, req_indices, accepted_steps)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return self.resource.spec.cpu_regions

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return self.resource.spec.pd_regions

    def _check_layout(self, layout):
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"

    def write_state_pages_to_cpu(
        self, page_kind, page_slot_ids, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None
    ):
        self._check_layout(layout)
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        assert isinstance(page_slot_ids, list) and isinstance(cpu_page_indices, list), "page indices must be lists"
        assert len(page_slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
        pool = self.resource.pool(page_kind)
        fields, pages = [], []
        for index, (slot, page) in enumerate(zip(page_slot_ids, cpu_page_indices)):
            if cpu_page_ready_flags is not None and cpu_page_ready_flags[index]:
                continue
            fields.append(self.resource.page_fields(pool, slot))
            pages.append(page)
        self.transfer.pages(
            fields=fields,
            page_buffers=cpu_page_buffers,
            page_indices=pages,
            layout=layout,
            write=True,
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        """先恢复到已预留的大页槽位；调用方另行 restore 才会更新请求运行态。"""
        self._check_layout(layout)
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        assert isinstance(big_page_slot_ids, list) and isinstance(cpu_page_indices, list), "page indices must be lists"
        assert len(big_page_slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        fields = [self.resource.page_fields(self.resource.pool("big"), slot) for slot in big_page_slot_ids]
        self.transfer.pages(
            fields=fields,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=False,
        )

    def _check_peer(self, peer, source_indices, destination_indices):
        assert isinstance(source_indices, list) and isinstance(destination_indices, list), "indices must be lists"
        assert len(source_indices) == len(destination_indices), "source and destination counts differ"
        assert (
            type(peer) is type(self) and peer.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        source = remote_dp_bounded_state
        self._check_peer(source, src_page_ids, dst_page_ids)
        if not self.resource.spec.snapshot_fields:
            return
        src_pool, dst_pool = source.resource.pool("big"), self.resource.pool("big")
        self.transfer.copy_fields(
            [source.resource.page_fields(src_pool, slot) for slot in src_page_ids],
            [self.resource.page_fields(dst_pool, slot) for slot in dst_page_ids],
        )

    def copy_runtime_state_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state):
        source = remote_dp_bounded_state
        self._check_peer(source, src_req_indices, dst_req_indices)
        self.runtime.check_dp_compatible(source.resource, self.resource)
        self.transfer.copy_fields(
            [source.runtime.dp_fields(source.resource, req) for req in src_req_indices],
            [self.runtime.dp_fields(self.resource, req) for req in dst_req_indices],
        )
        self.runtime.after_restore(self.resource, dst_req_indices)

    def write_to_pd_page(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
        self._check_layout(layout)
        self.transfer.pages(
            fields=[self.runtime.dp_fields(self.resource, req_idx)],
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=True,
        )

    def read_from_pd_page(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
        self._check_layout(layout)
        self.transfer.pages(
            fields=[self.runtime.dp_fields(self.resource, req_idx)],
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=False,
        )
        self.runtime.after_restore(self.resource, [req_idx])


class WindowBoundedState(ComposedBoundedState):
    """按绝对前缀终点搬运逻辑窗口，PD 表示不包含本地 MTP 扩展容量。"""

    def write_to_pd_page(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
        self._check_layout(layout)
        for external in self.transfer.external_views(
            page_buffers=pd_page_buffers,
            page_index=pd_page_index,
            layout=layout,
            write=True,
        ):
            self.runtime.copy_window(
                storage=self.resource,
                req_idx=req_idx,
                external=external,
                state_token_len=state_token_len,
                write=True,
            )

    def read_from_pd_page(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
        self._check_layout(layout)
        for external in self.transfer.external_views(
            page_buffers=pd_page_buffers,
            page_index=pd_page_index,
            layout=layout,
            write=False,
        ):
            self.runtime.copy_window(
                storage=self.resource,
                req_idx=req_idx,
                external=external,
                state_token_len=state_token_len,
                write=False,
            )
