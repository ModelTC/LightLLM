"""Select the local TP group and submit its State-owned PD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence
    from lightllm.common.att_resources.att_resource_manager import AttResourceManager
    from typing import Literal


def allocate_transfer_buffer(resources: AttResourceManager, page_count: int, *, device: str | torch.device | int):
    if page_count <= 0:
        raise ValueError("PD buffer must contain pages")
    return torch.empty((page_count, resources.get_pd_page_bytes()), dtype=torch.uint8, device=device)


def copy_page(
    node_att_managers: Sequence[AttResourceManager],
    pd_page_buffers: torch.Tensor,
    *,
    pd_page_index: int,
    dp_rank_in_node: int,
    tp_world_size: int,
    token_indices: torch.Tensor,
    task_kind: Literal["kv", "att_state"] = "kv",
    req_idx: int | None = None,
    state_token_len: int | None = None,
    write_to_pd_buffer: bool,
) -> None:
    """选择节点内指定 DP 的 TP 组，检查整组布局后提交 PD 页面读写。

    node_att_managers 按节点内 DP 编号、组内 TP rank 顺序平铺。
    pd_page_buffers 是 PD 页集合，pd_page_index 选择本次传输的页。
    write_to_pd_buffer=True 将 State 写入发送页；False 从接收页恢复 State。
    """
    tp_att_managers = node_att_managers[dp_rank_in_node * tp_world_size : (dp_rank_in_node + 1) * tp_world_size]
    assert (
        dp_rank_in_node >= 0
        and tp_world_size > 0
        and len(tp_att_managers) == tp_world_size
        and 0 <= pd_page_index < len(pd_page_buffers)
    ), "invalid PD group or page index"
    tp_transfer_layouts = []
    for att_manager in tp_att_managers:
        match task_kind:
            case "kv":
                state_layouts = att_manager.get_pd_full_att_transfer_layouts()
            case "att_state":
                state_layouts = att_manager.get_pd_bounded_att_or_tail_transfer_layouts()
            case _:
                assert False, f"unknown PD task kind {task_kind}"
        assert state_layouts and att_manager.get_pd_page_bytes() == pd_page_buffers.shape[1], (
            "PD page layout does not match this group"
        )
        tp_transfer_layouts.append(state_layouts)

    for att_manager, state_layouts in zip(tp_att_managers, tp_transfer_layouts):
        match task_kind:
            case "kv":
                if write_to_pd_buffer:
                    att_manager.write_to_pd_buffer(
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        kind=task_kind,
                        layouts=state_layouts,
                        token_indices=token_indices,
                    )
                else:
                    att_manager.read_from_pd_buffer(
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        kind=task_kind,
                        layouts=state_layouts,
                        token_indices=token_indices,
                    )
            case "att_state":
                if write_to_pd_buffer:
                    att_manager.write_to_pd_buffer(
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        kind=task_kind,
                        layouts=state_layouts,
                        req_idx=req_idx,
                        state_token_len=state_token_len,
                    )
                else:
                    att_manager.read_from_pd_buffer(
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        kind=task_kind,
                        layouts=state_layouts,
                        req_idx=req_idx,
                        state_token_len=state_token_len,
                    )
