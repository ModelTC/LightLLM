"""GDN replay 的已接受快照读写。

页布局在 replay_gdn_checkpoint。这里决定写哪一段 Conv、是否物化 SSM，以及何时清 history。
"""

from __future__ import annotations

import torch

from ..copy_kernel import copy_bounded_att_tensors
from ..transfer_utils import bounded_att_buffer_copies
from .replay_gdn_checkpoint import page_pool, tensor_fields


def accepted_conv(state, req_idx: int) -> torch.Tensor:
    """取出该请求已接受的 Conv 窗口，不包含 MTP 候选区。"""
    width = state.config.conv_width
    conv = state.conv_state[:, req_idx]
    if state.mtp_state_index is None:
        return conv[:, :, :width]
    offset = state.mtp_state_index[req_idx]
    return conv.index_select(-1, torch.arange(width, device=conv.device) + offset)


def save_runtime(state, req_indices, state_token_lens, page_kind, page_slot_ids) -> None:
    """把已接受 Conv 窗口和物化 SSM 写入页，不改 canonical state。"""
    assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
    assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
    pool = page_pool(state, page_kind)
    copy_bounded_att_tensors(
        copies=[
            (accepted_conv(state, req_idx), pool.conv_state[slot_id])
            for req_idx, slot_id in zip(req_indices, page_slot_ids)
        ]
    )
    state.runtime.export_ssm(req_indices, [pool.ssm_state[slot_id] for slot_id in page_slot_ids])


def restore_runtime(state, req_indices, state_token_lens, page_kind, page_slot_ids) -> None:
    """从页恢复基准 Conv/SSM，并清空目标请求的 history。"""
    assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
    assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
    pool = page_pool(state, page_kind)
    width = state.config.conv_width
    conv_copies, ssm_copies = [], []
    for req_idx, slot_id in zip(req_indices, page_slot_ids):
        state.runtime.clear_history(req_idx)
        conv_copies.append((pool.conv_state[slot_id], state.conv_state[:, req_idx, :, :width]))
        ssm_copies.append((pool.ssm_state[slot_id], state.ssm_state[:, req_idx]))
    copy_bounded_att_tensors(copies=conv_copies)
    copy_bounded_att_tensors(copies=ssm_copies)
    if state.mtp_state_index is not None:
        for req_idx in req_indices:
            state.mtp_state_index[req_idx].fill_(0)


def copy_runtime_from_dp(state, src_req_indices, dst_req_indices, remote_dp_bounded_state) -> None:
    """复制已接受运行态到目标单槽，并清空目标 history。"""
    assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
    assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
    assert (
        type(remote_dp_bounded_state) is type(state) and remote_dp_bounded_state.config == state.config
    ), "DP copy requires the same algorithm and equivalent TP state configurations"
    width = state.config.conv_width
    conv_copies = []
    for src_req, dst_req in zip(src_req_indices, dst_req_indices):
        state.runtime.clear_history(dst_req)
        conv_copies.append(
            (
                accepted_conv(remote_dp_bounded_state, src_req),
                state.conv_state[:, dst_req, :, :width],
            )
        )
    copy_bounded_att_tensors(copies=conv_copies)
    remote_dp_bounded_state.runtime.export_ssm(
        src_req_indices, [state.ssm_state[:, dst_req] for dst_req in dst_req_indices]
    )
    if state.mtp_state_index is not None:
        for dst_req in dst_req_indices:
            state.mtp_state_index[dst_req].fill_(0)


def write_pd(state, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout):
    """写出已接受 Conv 窗口和物化 SSM，不改写源 history。"""
    assert layout.state_name == state.state_name, "transfer layout does not describe this resource"
    assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
    accepted = state.get_accepted_state(req_idx)
    fields = tensor_fields(accepted.conv, accepted.ssm)
    for copies in bounded_att_buffer_copies(
        page_fields=[fields],
        page_buffers=pd_page_buffers,
        layout=layout,
        page_indices=[pd_page_index],
        write=True,
    ):
        copy_bounded_att_tensors(copies=copies)


def read_pd(state, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout):
    """读入基准 Conv/SSM，并清空该请求的 history。"""
    assert layout.state_name == state.state_name, "transfer layout does not describe this resource"
    assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
    state.runtime.clear_history(req_idx)
    fields = tensor_fields(
        state.conv_state[:, req_idx, :, : state.config.conv_width],
        state.ssm_state[:, req_idx],
    )
    for copies in bounded_att_buffer_copies(
        page_fields=[fields],
        page_buffers=pd_page_buffers,
        layout=layout,
        page_indices=[pd_page_index],
        write=False,
    ):
        copy_bounded_att_tensors(copies=copies)
    if state.mtp_state_index is not None:
        state.mtp_state_index[req_idx].fill_(0)
