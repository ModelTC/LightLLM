"""NoPE MLA with one independently aligned indexer entry per four tokens.

The allocator owns complete pages divisible by four. A closed logical pool's
last token therefore identifies its physical indexer slot as token_slot // 4.
Incomplete pools live in the bounded raw-tail state, not in this indexer.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from ...types import AxisMapping, TransferRegion
from ...utils import deepcopy_result
from ..base_state import FullAttentionState
from ..dp_transfer import copy_packed_tokens
from .packed_mla_state import PackedMLALayerView
from .packed_transfer import transfer_packed_pages


@dataclass(frozen=True)
class PooledMLAConfig:
    logical_layer_groups: tuple[tuple[int, ...], ...]
    format: str
    latent_dim: int = 512
    indexer_dim: int = 128
    tp_rank: int = 0
    tp_world_size: int = 1

    def __post_init__(self):
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert all(isinstance(layer, int) and layer >= 0 for layer in layers)
        assert len(layers) == len(set(layers))
        assert self.latent_dim == 512 and self.indexer_dim == 128
        assert self.format in ("packed_bf16", "packed_fp16", "packed_group64_e8m0", "packed_nvfp4")
        assert 0 <= self.tp_rank < self.tp_world_size

    @property
    def dtype(self):
        return {"packed_bf16": torch.bfloat16, "packed_fp16": torch.float16}.get(self.format, torch.uint8)

    @property
    def kv_bytes(self):
        return {"packed_nvfp4": 288, "packed_group64_e8m0": 584}.get(self.format, self.latent_dim * 2)

    @property
    def kv_stride_bytes(self):
        return (self.kv_bytes + 15) // 16 * 16

    @property
    def indexer_stride_bytes(self):
        return (self.indexer_dim + 4 + 15) // 16 * 16

    @property
    def token_bytes(self):
        return self.kv_stride_bytes + self.indexer_stride_bytes // 4


class PooledMLAState(FullAttentionState):
    def __init__(self, layout_config: PooledMLAConfig, page_size=4, device="cuda"):
        assert page_size > 0 and page_size % 4 == 0, "pooled MLA requires --page_size divisible by 4"
        self.config = layout_config
        self.state_name = None
        self.page_size = page_size
        self.device = torch.device(device)
        self.capacity = 0
        self.storage = None
        self.indexer_storage = None
        self._logical_to_storage_layer = {
            layer: index for index, group in enumerate(layout_config.logical_layer_groups) for layer in group
        }

    def get_state_type(self):
        return "mla_pooled4_" + self.config.format.removeprefix("packed_")

    def allocate_storage(self, token_capacity):
        assert token_capacity >= 0 and token_capacity % self.page_size == 0
        self.capacity = token_capacity
        layers = len(self.config.logical_layer_groups)
        tokens = token_capacity + self.page_size
        self.storage = torch.empty(
            (layers, tokens, 1, self.config.kv_stride_bytes // self.config.dtype.itemsize),
            dtype=self.config.dtype,
            device=self.device,
        )
        self.indexer_storage = torch.empty(
            (layers, tokens // 4, 1, self.config.indexer_stride_bytes), dtype=torch.uint8, device=self.device
        )

    def get_token_bytes(self):
        return len(self.config.logical_layer_groups) * self.config.token_bytes

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id):
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id):
        layer = self.get_storage_layer_index(logical_layer_id)
        return PackedMLALayerView(
            kv=self.storage[layer, ..., : self.config.kv_bytes // self.config.dtype.itemsize],
            indexer_k=self.indexer_storage[layer, ..., : self.config.indexer_dim + 4],
            indexer_token_stride=4,
        )

    def write_full_att_kv(self, logical_layer_id, token_indices, kv):
        if not token_indices.numel():
            return
        target = self.get_layer_view(logical_layer_id).kv
        if self.config.format == "packed_nvfp4":
            from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla

            assert kv.dtype == torch.bfloat16
            quantize_nvfp4_mla(kv, token_indices, target)
        elif self.config.format == "packed_group64_e8m0":
            from .fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv

            write_fp8_e8m0_packed_kv(kv, token_indices, target)
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            assert kv.dtype == self.config.dtype
            destindex_copy_kv(kv, token_indices, target)

    def copy_full_att_tokens(self, src_indices, dst_indices):
        self.copy_tokens_from_dp(src_indices, dst_indices, self)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        remote = remote_dp_full_att_state
        assert type(remote) is type(self) and remote.config == self.config
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda
        assert src_token_indices.ndim == dst_token_indices.ndim == 1
        assert src_token_indices.numel() == dst_token_indices.numel()
        # Request-relative positions are preserved by token/DP migration. Copy
        # indexer entries only at pool-closing slots, including partial tails.
        for source, destination, divisor in (
            (remote.storage, self.storage, 1),
            (remote.indexer_storage, self.indexer_storage, 4),
        ):
            copy_packed_tokens(source, destination, src_token_indices, dst_token_indices, token_divisor=divisor)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().cpu_cache_token_page_size)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return self._get_transfer_regions(get_env_start_args().pd_kv_page_size)

    def _get_transfer_regions(self, page_tokens):
        assert page_tokens > 0 and page_tokens % 4 == 0, "pooled MLA transfer pages must align to four tokens"
        regions, offset = [], 0
        for field, tokens, width, dtype in (
            ("kv", page_tokens, self.config.kv_bytes // self.config.dtype.itemsize, self.config.dtype),
            ("indexer_k", page_tokens // 4, self.config.indexer_dim + 4, torch.uint8),
        ):
            shape = (len(self.config.logical_layer_groups), tokens, 1, width)
            end = offset + prod(shape) * dtype.itemsize
            regions.append(
                TransferRegion(
                    field=field,
                    byte_start=offset,
                    byte_end=end,
                    dtype=dtype,
                    axes=("layer", "token", "head", "dim"),
                    shape=shape,
                    axis_mappings=(
                        AxisMapping(axis="head", local_start=0, external_start=0, length=1),
                        AxisMapping(axis="dim", local_start=0, external_start=0, length=width),
                    ),
                    write_enabled=self.config.tp_rank == 0,
                )
            )
            offset = end
        return tuple(regions)

    def _transfer(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None):
        transfer_packed_pages(
            kv_storage=self.storage[..., : self.config.kv_bytes // self.config.dtype.itemsize],
            indexer_storage=self.indexer_storage[..., : self.config.indexer_dim + 4],
            indexer_token_stride=4,
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
