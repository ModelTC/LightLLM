from __future__ import annotations
from typing import List
import torch
from lightllm.common.attention_state.types import FieldName
from lightllm.common.attention_state.types import PageKind
from lightllm.common.attention_state.bounded_att.copy_kernel import copy_bounded_att_tensors
from .swa_snapshot import copy_swa_window
from .compressor_snapshot import copy_compressor_window


class BoundedRuntime:
    """解释请求状态的时间语义和候选位置，不拥有 Tensor 或页池槽位。

    save/restore 使用已执行的绝对 token 长度；不同算法的快照边界由实现规定。
    涉及 CUDA 的操作异步提交，调用方负责流顺序与存储存活。
    """

    def __init__(self, config):
        self.config = config

    def prepare(self, storage):
        pass

    def after_restore(self, storage, req_indices):
        pass

    def check_dp_compatible(self, source, destination):
        assert source.mtp_step == destination.mtp_step, "DP ring copy requires equal MTP capacity"


class ConvSSMRuntime(BoundedRuntime):
    """管理 Conv 历史与 SSM 候选；checkpoint/PD 读取基准候选 0。

    调用方须保证基准候选确实对应声明的前缀终点，不能将 MTP 游标提交
    误认为已把接受候选复制回基准位置。
    """

    def layer_view(self, storage, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号映射到物理层，返回 Conv/SSM Tensor 引用；MTP 游标和步数直接从 State 读取。"""
        storage_layer_index = storage.layer_indices[logical_layer_id]
        return (
            storage.tensor(FieldName.CONV_STATE)[storage_layer_index],
            storage.tensor(FieldName.SSM_STATE)[storage_layer_index],
        )

    def reset(self, storage, req_indices: List[int]) -> None:
        """清零指定请求的 Conv 历史、全部 SSM 候选及 MTP 游标，保留页池快照。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            storage.tensor(FieldName.CONV_STATE)[:, req_idx].zero_()
            start = req_idx * (storage.mtp_step + 1)
            storage.tensor(FieldName.SSM_STATE)[:, start : start + storage.mtp_step + 1].zero_()
            if storage.buffers.get(FieldName.MTP_STATE_INDEX) is not None:
                storage.buffers.get(FieldName.MTP_STATE_INDEX)[req_idx].fill_(0)

    def save(
        self,
        storage,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """保存基准 Conv/SSM；不会按 MTP 游标选择候选，长度标签由调用方保证。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = storage.pool(page_kind)
        conv_copies, ssm_copies = ([], [])
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append(
                (
                    storage.tensor(FieldName.CONV_STATE)[:, req_idx, :, : self.config.conv_width],
                    pool.conv_state[slot_id],
                )
            )
            ssm_copies.append(
                (storage.tensor(FieldName.SSM_STATE)[:, req_idx * (storage.mtp_step + 1)], pool.ssm_state[slot_id])
            )
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def restore(
        self,
        storage,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """恢复各请求指定绝对 token 长度处的基准状态并归零游标；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = storage.pool(page_kind)
        conv_copies, ssm_copies = ([], [])
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append(
                (
                    pool.conv_state[slot_id],
                    storage.tensor(FieldName.CONV_STATE)[:, req_idx, :, : self.config.conv_width],
                )
            )
            ssm_copies.append(
                (pool.ssm_state[slot_id], storage.tensor(FieldName.SSM_STATE)[:, req_idx * (storage.mtp_step + 1)])
            )
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if storage.buffers.get(FieldName.MTP_STATE_INDEX) is not None:
            for req_idx in req_indices:
                storage.buffers.get(FieldName.MTP_STATE_INDEX)[req_idx].fill_(0)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """用 CUDA 请求索引和 int32 接受步数选择 SSM 候选位置；步数包含基准输入步。"""
        assert req_indices.is_cuda and accepted_steps.is_cuda, "commit inputs must be CUDA tensors"
        if storage.buffers.get(FieldName.MTP_STATE_INDEX) is not None:
            storage.buffers.get(FieldName.MTP_STATE_INDEX)[req_indices] = accepted_steps - 1

    def dp_fields(self, storage, req: int):
        return {
            FieldName.CONV: storage.bind_field(
                FieldName.CONV, storage.tensor(FieldName.CONV_STATE)[:, req, :, : self.config.conv_width]
            ),
            FieldName.SSM: storage.bind_field(
                FieldName.SSM, storage.tensor(FieldName.SSM_STATE)[:, req * (storage.mtp_step + 1)]
            ),
        }

    def after_restore(self, storage, req_indices):
        cursor = storage.buffers.get(FieldName.MTP_STATE_INDEX)
        if cursor is not None:
            for req in req_indices:
                cursor[req].fill_(0)

    def check_dp_compatible(self, source, destination):
        pass


class SlidingWindowRuntime(BoundedRuntime):
    """在物理 ring 与按时间排序的逻辑快照之间转换，MTP 扩展只属于本地容量。"""

    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """返回该层的 [request, ring_token, packed_byte] 视图。"""
        return storage.tensor(FieldName.SWA_BUFFER)[storage.layer_indices[logical_layer_id]]

    def reset(self, storage, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            storage.tensor(FieldName.SWA_BUFFER)[:, req_idx].zero_()

    def save(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=True)

    def restore(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=False)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """ring 按绝对 token 位置寻址，无需提交额外游标。"""
        pass

    def copy_window(self, storage, req_idx: int, external: torch.Tensor, state_token_len: int, write: bool) -> None:
        copy_swa_window(
            local=storage.tensor(FieldName.SWA_BUFFER)[:, req_idx],
            external=external,
            state_token_len=state_token_len,
            config=self.config,
            write=write,
        )

    def dp_fields(self, storage, req):
        return {FieldName.SWA: storage.bind_field(FieldName.SWA, storage.tensor(FieldName.SWA_BUFFER)[:, req])}


class CompressorRuntime(BoundedRuntime):
    """解释 candidate|gate 窗口；CPU 快照仅在完整产出步长边界保存重叠历史。"""

    def layer_view(self, storage, logical_layer_id: int) -> torch.Tensor:
        """返回该层 [request, window+MTP, candidate|gate] 的计算视图。"""
        return storage.tensor(FieldName.BUFFER)[storage.layer_indices[logical_layer_id]]

    def reset(self, storage, req_indices: list[int]) -> None:
        for req_idx in req_indices:
            storage.tensor(FieldName.BUFFER)[:, req_idx].zero_()
            storage.tensor(FieldName.BUFFER)[:, req_idx, :, self.config.candidate_dim :].fill_(float("-inf"))

    def save(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        """边界必须完成一个产出步长；仅保留下一步仍需读取的窗口重叠部分。"""
        assert all(
            (length % self.config.compress_stride == 0 for length in state_token_lens)
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=True)

    def restore(
        self,
        storage,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind: PageKind,
        page_slot_ids: list[int],
    ) -> None:
        assert all(
            (length % self.config.compress_stride == 0 for length in state_token_lens)
        ), "prefix must close a compressor stride"
        if not self.config.overlap_tokens:
            return
        pool = storage.pool(page_kind=page_kind)
        for req_idx, length, slot in zip(req_indices, state_token_lens, page_slot_ids):
            self.copy_window(storage, req_idx=req_idx, external=pool.buffer[slot], state_token_len=length, write=False)

    def commit(self, storage, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """按绝对位置寻址；接受后继续覆盖 ring，不维护额外 MTP 游标。"""
        pass

    def copy_window(self, storage, req_idx: int, external: torch.Tensor, state_token_len: int, write: bool) -> None:
        copy_compressor_window(
            local=storage.tensor(FieldName.BUFFER)[:, req_idx],
            external=external,
            state_token_len=state_token_len,
            write=write,
        )

    def dp_fields(self, storage, req):
        return {
            FieldName.COMPRESSOR: storage.bind_field(FieldName.COMPRESSOR, storage.tensor(FieldName.BUFFER)[:, req])
        }

    def prepare(self, storage):
        storage.tensor(FieldName.BUFFER)[:, storage.request_capacity - 1, :, self.config.candidate_dim :].fill_(
            float("-inf")
        )
