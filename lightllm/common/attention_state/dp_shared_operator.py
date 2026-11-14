from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from lightllm.common.attention_state.manager import AttentionStateManager


class DPSharedOperator:
    """封装 DP shared 的跨副本复制、IPC 导出和共享页注册。"""

    def __init__(self, owner: AttentionStateManager):
        self.owner = owner

    def create_ipc_copy(self) -> AttentionStateManager:
        """保留共享运行态和共享 CPU 页池，排除本地 allocator。"""
        for state in self.owner.get_bounded_attention_states():
            for pool in (state.big_pages_pool, state.small_pages_pool):
                if pool is not None:
                    for tensor in vars(pool).values():
                        if isinstance(tensor, torch.Tensor):
                            assert (
                                tensor.device.type == "cpu" and tensor.is_shared()
                            ), "DP IPC requires shared CPU page pools"
        ipc_copy = copy(self.owner)
        ipc_copy.full_att_allocator = None
        ipc_copy.bounded_att_big_page_allocator = None
        ipc_copy.bounded_att_small_page_allocator = None
        ipc_copy._reset_operators()
        return ipc_copy

    def register_page_pools_pin_memory(self) -> None:
        """在当前 GPU 映射所有共享 bounded 页池。"""
        for state in self.owner.get_bounded_attention_states():
            if state.big_pages_pool is not None:
                state.big_pages_pool.register_pin_memory()
            if state.small_pages_pool is not None:
                state.small_pages_pool.register_pin_memory()

    def copy_from_dp(
        self,
        remote_att_manager: AttentionStateManager,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        src_big_page_ids: List[int],
        dst_big_page_ids: List[int],
        src_req_idx: int,
        dst_req_idx: int,
    ) -> None:
        """复制其他 DP 的 token、大小页和 prefill 基准运行态。"""
        self.copy_batch_from_dp(
            remote_att_manager=remote_att_manager,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
            src_big_page_ids=src_big_page_ids,
            dst_big_page_ids=dst_big_page_ids,
            src_req_indices=[src_req_idx],
            dst_req_indices=[dst_req_idx],
        )

    def copy_batch_from_dp(
        self,
        remote_att_manager: AttentionStateManager,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        src_big_page_ids: List[int],
        dst_big_page_ids: List[int],
        src_req_indices: List[int],
        dst_req_indices: List[int],
    ) -> None:
        assert type(remote_att_manager) is type(self.owner), "DP replicas must use the same model resource manager"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        assert len(src_big_page_ids) == len(dst_big_page_ids), "source and destination page counts differ"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        if src_token_indices.numel():
            for state in self.owner.get_full_attention_states():
                state.copy_tokens_from_dp(
                    src_token_indices=src_token_indices,
                    dst_token_indices=dst_token_indices,
                    remote_dp_full_att_state=getattr(remote_att_manager, state.state_name),
                )
        if src_big_page_ids:
            for state in self.owner.get_bounded_attention_states():
                state.copy_big_pages_from_dp(
                    src_page_ids=src_big_page_ids,
                    dst_page_ids=dst_big_page_ids,
                    remote_dp_bounded_state=getattr(remote_att_manager, state.state_name),
                )
        for state in self.owner.get_bounded_attention_states():
            state.copy_runtime_state_from_dp(
                src_req_indices=src_req_indices,
                dst_req_indices=dst_req_indices,
                remote_dp_bounded_state=getattr(remote_att_manager, state.state_name),
            )
