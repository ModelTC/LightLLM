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
    resources: Sequence[AttResourceManager],
    buffer: torch.Tensor,
    *,
    page_index: int,
    dp_index: int,
    tp_size: int,
    token_indices: torch.Tensor,
    kind: Literal["kv", "att_state"] = "kv",
    req_idx: int | None = None,
    state_token_len: int | None = None,
    write: bool,
):
    ranks = resources[dp_index * tp_size : (dp_index + 1) * tp_size]
    if dp_index < 0 or tp_size <= 0 or len(ranks) != tp_size or not 0 <= page_index < len(buffer):
        raise ValueError("invalid PD group or page index")
    rank_layouts = []
    for resource in ranks:
        if kind == "kv":
            layouts = resource.get_pd_full_att_transfer_layouts()
        elif kind == "att_state":
            layouts = resource.get_pd_bounded_att_or_tail_transfer_layouts()
        else:
            raise ValueError(f"unknown PD task kind {kind}")
        if not layouts or resource.get_pd_page_bytes() != buffer.shape[1]:
            raise ValueError("PD page layout does not match this group")
        rank_layouts.append(layouts)
    for resource, layouts in zip(ranks, rank_layouts):
        kwargs = {"kind": kind, "layouts": layouts}
        if kind == "kv":
            kwargs["token_indices"] = token_indices
        else:
            kwargs["req_idx"] = req_idx
            kwargs["state_token_len"] = state_token_len
        if write:
            resource.write_to_pd_buffer(buffer, pd_page_index=page_index, **kwargs)
        else:
            resource.read_from_pd_buffer(buffer, pd_page_index=page_index, **kwargs)
