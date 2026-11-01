import torch
from ...types import StateTransferLayout, TensorField
from ..page_transfer import transfer_token_pages


def transfer_gqa_pages(
    k_storage: torch.Tensor,
    v_storage: torch.Tensor,
    token_indices: torch.Tensor,
    page_buffers: torch.Tensor,
    page_indices: list[int],
    layout: StateTransferLayout,
    write: bool,
    grid_num: int = 24,
    page_readies: list[bool] | None = None,
    num_warps: int = 8,
    num_stages: int = 1,
) -> None:
    transfer_token_pages(
        fields={
            "k": TensorField(k_storage, ("layer", "token", "head", "dim")),
            "v": TensorField(v_storage, ("layer", "token", "head", "dim")),
        },
        token_indices=token_indices,
        page_buffers=page_buffers,
        page_indices=page_indices,
        layout=layout,
        write=write,
        grid_num=grid_num,
        page_readies=page_readies,
        num_warps=num_warps,
        num_stages=num_stages,
    )
