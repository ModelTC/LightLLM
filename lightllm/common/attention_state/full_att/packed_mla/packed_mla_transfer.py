import torch
from ...types import StateTransferLayout, FieldName
from .formats import PackedMLALogicalConfig, PackedMLAAttentionConfig
from ..page_transfer import transfer_token_pages


def transfer_packed_mla_pages(
    kv_storage: torch.Tensor,
    token_indices: torch.Tensor,
    page_buffers: torch.Tensor,
    page_indices: list[int],
    layout: StateTransferLayout,
    kv_dim: int,
    indexer_offset: int,
    indexer_dim: int,
    write: bool,
    grid_num: int = 24,
    page_readies: list[bool] | None = None,
    num_warps: int = 8,
    num_stages: int = 1,
) -> None:
    # 兼容旧入口的参数；字节切片仍由同一个格式规格生成。
    encoding = {torch.uint8: "packed_group128_fp8", torch.float16: "packed_fp16", torch.bfloat16: "packed_bf16"}
    config = PackedMLAAttentionConfig(
        PackedMLALogicalConfig(
            logical_layer_groups=tuple((i,) for i in range(kv_storage.shape[0])),
            latent_dim=kv_dim,
            format=encoding[kv_storage.dtype],
            indexer_dim=indexer_dim,
        )
    )
    assert indexer_offset == config.indexer_offset
    fields = config.storage_spec.page_views({FieldName.KV: kv_storage})
    transfer_token_pages(
        fields=fields,
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
