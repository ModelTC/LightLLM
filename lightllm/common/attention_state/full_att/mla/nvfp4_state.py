"""Replicated NVFP4 MLA latent, optionally packed with a sparse indexer.

Each block of 16 latent values has eight E2M1 bytes and one E4M3 scale.
RoPE, when present, retains the computation dtype. Indexer keys and their
FP32 scales retain their existing encoding. Transfers include all scales.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from lightllm.utils.envs_utils import get_env_start_args
from ...types import AxisMapping, TransferRegion
from ...utils import deepcopy_result
from ..base_state import FullAttentionState
from ..dp_transfer import copy_packed_tokens
from ..packed_mla.packed_mla_state import PackedMLALayerView
from ..packed_mla.packed_transfer import transfer_packed_pages


@dataclass(frozen=True)
class NVFP4MLAConfig:
    logical_layer_groups: tuple[tuple[int, ...], ...]
    latent_dim: int = 512
    rope_dim: int = 0
    indexer_dim: int = 0
    dtype: torch.dtype = torch.bfloat16
    tp_rank: int = 0
    tp_world_size: int = 1

    def __post_init__(self):
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert self.latent_dim > 0 and self.latent_dim % 32 == 0
        assert self.rope_dim >= 0 and self.rope_dim % 16 == 0 and self.indexer_dim >= 0
        assert self.dtype in (torch.bfloat16, torch.float16)
        assert not self.indexer_dim or (self.kv_bytes + self.indexer_dim) % 4 == 0, "indexer scale must be FP32 aligned"
        assert 0 <= self.tp_rank < self.tp_world_size

    @property
    def kv_bytes(self):
        return self.latent_dim // 2 + self.latent_dim // 16 + self.rope_dim * 2

    @property
    def payload_bytes(self):
        return self.kv_bytes + (self.indexer_dim + 4 if self.indexer_dim else 0)

    @property
    def token_bytes(self):
        return (self.payload_bytes + 15) // 16 * 16


class NVFP4MLAState(FullAttentionState):
    def __init__(self, layout_config: NVFP4MLAConfig, page_size=1, device="cuda"):
        self.config = layout_config
        self.state_name = None
        self.page_size = page_size
        self.device = torch.device(device)
        self.capacity = 0
        self.storage = None
        self._logical_to_storage_layer = {
            layer: index for index, group in enumerate(layout_config.logical_layer_groups) for layer in group
        }

    def get_state_type(self):
        suffix = "_packed" if self.config.indexer_dim else ""
        return "mla_nvfp4_block16" + suffix

    def allocate_storage(self, token_capacity):
        self.capacity = token_capacity
        self.storage = torch.empty(
            (len(self.config.logical_layer_groups), token_capacity + self.page_size, 1, self.config.token_bytes),
            dtype=torch.uint8,
            device=self.device,
        )

    def get_token_bytes(self):
        return len(self.config.logical_layer_groups) * self.config.token_bytes

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id):
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id):
        row = self.storage[self.get_storage_layer_index(logical_layer_id)]
        kv = row[..., : self.config.kv_bytes]
        if not self.config.indexer_dim:
            return kv
        return PackedMLALayerView(kv=kv, indexer_k=row[..., self.config.kv_bytes : self.config.payload_bytes])

    def write_full_att_kv(self, logical_layer_id, token_indices, kv):
        from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla

        assert kv.dtype == self.config.dtype, "MLA KV writes must match the configured computation dtype"
        row = self.storage[self.get_storage_layer_index(logical_layer_id)]
        quantize_nvfp4_mla(kv, token_indices, row[..., : self.config.kv_bytes], self.config.rope_dim)

    def copy_full_att_tokens(self, src_indices, dst_indices):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        assert src_indices.is_cuda and dst_indices.is_cuda
        assert src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel()
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (torch.int32, torch.int64)
        if src_indices.numel():
            copy_kv_buffer_to_kv_buffer(src_mem_index=src_indices, dst_mem_index=dst_indices, kv_buffer=self.storage)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().cpu_cache_token_page_size)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().pd_kv_page_size)

    def _get_transfer_regions(self, page_tokens):
        shape = (len(self.config.logical_layer_groups), page_tokens, 1, self.config.payload_bytes)
        return (
            TransferRegion(
                field="kv",
                byte_start=0,
                byte_end=shape[0] * page_tokens * self.config.payload_bytes,
                dtype=torch.uint8,
                axes=("layer", "token", "head", "dim"),
                shape=shape,
                axis_mappings=(
                    AxisMapping(axis="head", local_start=0, external_start=0, length=1),
                    AxisMapping(axis="dim", local_start=0, external_start=0, length=self.config.payload_bytes),
                ),
                write_enabled=self.config.tp_rank == 0,
            ),
        )

    def _transfer(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None):
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.payload_bytes],
            token_indices=token_indices,
            page_buffers=page_buffers,
            page_indices=page_indices,
            layout=layout,
            write=write,
            page_readies=page_readies,
        )

    def write_to_cpu(self, token_indices, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None):
        assert cpu_page_buffers.device.type == "cpu"
        self._transfer(token_indices, cpu_page_buffers, cpu_page_indices, layout, True, cpu_page_ready_flags)

    def read_from_cpu(self, cpu_page_buffers, cpu_page_indices, token_indices, layout):
        assert cpu_page_buffers.device.type == "cpu"
        self._transfer(token_indices, cpu_page_buffers, cpu_page_indices, layout, False)

    def write_to_pd_page(self, token_indices, pd_page_buffers, pd_page_index, layout):
        assert pd_page_buffers.is_cuda
        self._transfer(token_indices, pd_page_buffers, [pd_page_index], layout, True)

    def read_from_pd_page(self, pd_page_buffers, token_indices, pd_page_index, layout):
        assert pd_page_buffers.is_cuda
        self._transfer(token_indices, pd_page_buffers, [pd_page_index], layout, False)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda
        assert src_token_indices.numel() == dst_token_indices.numel()
        copy_packed_tokens(
            src_storage=remote_dp_full_att_state.storage,
            dst_storage=self.storage,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
        )
