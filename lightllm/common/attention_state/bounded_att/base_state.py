"""有界 attention 状态接口；具体算法实现运行态与大小页存储。"""

from __future__ import annotations

import torch

from .interface import CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface


class BoundedAttentionState(CoreInterface, CpuCacheInterface, DpCopyInterface, PdInterface):
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

    def allocate_runtime_storage(self, device):
        allocated = False
        for name in self.runtime_storage_fields:
            value = getattr(self, name)
            if value is not None and value.is_meta:
                setattr(self, name, torch.zeros_like(value, device=device))
                allocated = True
        return allocated

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

    def get_runtime_copy_offsets(self, requests, device):
        return self.get_runtime_transfer_view().offsets(requests, device)

    def finish_runtime_restore(self, requests):
        self.get_runtime_transfer_view().finish_restore(requests)

    def bind_pd_copy(self, pages, layout, write):
        from .copy_plan import BoundedPageCopy

        return BoundedPageCopy.bind(self.get_runtime_transfer_view(), pages, layout, write).run_pd

    def write_to_pd_page(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
        self.bind_pd_copy(pd_page_buffers, layout, True)(
            req_idx=req_idx, pd_page_index=pd_page_index, state_token_len=state_token_len
        )

    def read_from_pd_page(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
        self.bind_pd_copy(pd_page_buffers, layout, False)(
            req_idx=req_idx, pd_page_index=pd_page_index, state_token_len=state_token_len
        )

    def bind_cpu_copy(self, pages, layout, page_kind, write):
        from .copy_plan import BoundedPageCopy

        return BoundedPageCopy.bind(self.get_snapshot_transfer_view(page_kind), pages, layout, write)

    def write_state_pages_to_cpu(
        self, page_kind, page_slot_ids, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None
    ):
        self.bind_cpu_copy(cpu_page_buffers, layout, page_kind, True).run(
            page_slot_ids, cpu_page_indices, cpu_page_ready_flags
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        self.bind_cpu_copy(cpu_page_buffers, layout, "big", False).run(big_page_slot_ids, cpu_page_indices)

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        from .peer_copy import BoundedPeerCopy

        BoundedPeerCopy.bind(remote_dp_bounded_state, self).copy_pages(src_page_ids, dst_page_ids)

    def copy_runtime_state_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state):
        from .peer_copy import BoundedPeerCopy

        BoundedPeerCopy.bind(remote_dp_bounded_state, self).copy_runtime(src_req_indices, dst_req_indices)

    def release_page_pools(self):

        self.big_pages_pool = None
        self.small_pages_pool = None

    def get_checkpoint_bytes(self):
        if not self.get_cpu_transfer_regions():
            return 0
        raise NotImplementedError("state must declare the native size of one checkpoint")
