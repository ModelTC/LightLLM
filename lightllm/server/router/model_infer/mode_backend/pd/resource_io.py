"""Select the local TP group and submit its State-owned PD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
from lightllm.common.attention_state.copy_plan import CopyPlan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from lightllm.common.attention_state.manager import AttentionStateManager
    from typing import Literal


def allocate_transfer_buffer(resources: AttentionStateManager, page_count: int, device: str | torch.device | int):
    if page_count <= 0:
        raise ValueError("PD buffer must contain pages")
    return torch.empty((page_count, resources.get_pd_operator().get_page_bytes()), dtype=torch.uint8, device=device)


def copy_page(
    node_att_managers: Sequence[AttentionStateManager],
    pd_page_buffers: torch.Tensor,
    pd_page_index: int,
    dp_rank_in_node: int,
    tp_world_size: int,
    token_indices: torch.Tensor | None,
    write_to_pd_page: bool,
    task_kind: Literal["kv", "att_state"] = "kv",
    req_idx: int | None = None,
    state_token_len: int | None = None,
) -> None:
    """选择节点内指定 DP 的 TP 组，检查整组布局后提交 PD 页面读写。

    node_att_managers 按节点内 DP 编号、组内 TP rank 顺序平铺。
    pd_page_buffers 是 PD 页集合，pd_page_index 选择本次传输的页。
    token_indices 在 kv 任务中为 CUDA 索引；att_state 任务传 None，不进行索引上传。
    write_to_pd_page=True 将 State 写入发送页；False 从接收页恢复 State。
    """
    tp_att_managers = node_att_managers[dp_rank_in_node * tp_world_size : (dp_rank_in_node + 1) * tp_world_size]
    assert (
        dp_rank_in_node >= 0
        and tp_world_size > 0
        and len(tp_att_managers) == tp_world_size
        and 0 <= pd_page_index < len(pd_page_buffers)
    ), "invalid PD group or page index"
    CopyPlan.for_pd(tp_att_managers, pd_page_buffers, write_to_pd_page).run(
        pd_page_index=pd_page_index,
        task_kind=task_kind,
        token_indices=token_indices,
        req_idx=req_idx,
        state_token_len=state_token_len,
    )
