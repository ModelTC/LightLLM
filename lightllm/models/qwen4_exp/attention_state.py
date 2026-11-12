from dataclasses import replace

import torch

from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAFullAttentionState, GQAAttentionConfig
from .state import QSAExecutionState


class QSAIndexerState(GQAFullAttentionState):
    def __init__(self, layout_config, page_size, device, pooled_dtype):
        super().__init__(layout_config, page_size, device)
        self.pooled_dtype = pooled_dtype

    def allocate_storage(self, token_capacity):
        self.capacity = token_capacity
        self.k_storage = torch.empty(self._storage_shape(), dtype=self.storage_dtype, device=self.device)
        self.v_storage = torch.empty(self._storage_shape(), dtype=self.pooled_dtype, device=self.device)

    def get_token_bytes(self):
        config = self.config
        return (
            len(config.logical_layer_groups)
            * config.head_num
            * config.head_dim
            * (self.storage_dtype.itemsize + self.pooled_dtype.itemsize)
        )

    def _get_transfer_regions(self, page_tokens):
        key, value = super()._get_transfer_regions(page_tokens)
        value = replace(
            value,
            dtype=self.pooled_dtype,
            byte_end=value.byte_start
            + (value.byte_end - value.byte_start) * self.pooled_dtype.itemsize // self.storage_dtype.itemsize,
        )
        return key, value

    def copy_full_att_tokens(self, src_indices, dst_indices):
        from lightllm.common.attention_state.full_att.dp_transfer import copy_gqa_tokens

        copy_gqa_tokens(self.k_storage, self.v_storage, self.k_storage, self.v_storage, src_indices, dst_indices)


class QSAPooledState(GQAFullAttentionState):
    def allocate_storage(self, token_capacity):
        self.capacity = token_capacity
        self.k_storage = torch.empty(self._storage_shape(), dtype=self.storage_dtype, device=self.device)

    def get_token_bytes(self):
        return super().get_token_bytes() // 2

    def _get_transfer_regions(self, page_tokens):
        return super()._get_transfer_regions(page_tokens)[:1]

    def copy_full_att_tokens(self, src_indices, dst_indices):
        from lightllm.common.attention_state.full_att.dp_transfer import copy_packed_tokens

        copy_packed_tokens(self.k_storage, self.k_storage, src_indices, dst_indices)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        from lightllm.common.attention_state.full_att.dp_transfer import copy_packed_tokens

        copy_packed_tokens(remote_dp_full_att_state.k_storage, self.k_storage, src_token_indices, dst_token_indices)


class Qwen4ExpAttentionState(GQAFullAttentionState):
    def __init__(self, layout_config, indexer_dim, rotary_dim, page_size=1, device="meta", fp8=False, compress_ratio=1):
        super().__init__(layout_config, page_size, device)
        if compress_ratio < 1 or page_size % compress_ratio or indexer_dim % compress_ratio:
            raise ValueError("QSA compact cache requires complete aligned blocks and divisible indexer dimensions")
        self.compress_ratio = compress_ratio
        self.indexer_dim = indexer_dim
        if fp8:
            self.storage_dtype = torch.float8_e4m3fn
        common = dict(
            global_head_num=1,
            dtype=layout_config.dtype,
            tp_rank=layout_config.tp_rank,
            tp_world_size=layout_config.tp_world_size,
            replicated=True,
        )
        if compress_ratio > 1:
            self.indexer = QSAPooledState(
                GQAAttentionConfig(
                    logical_layer_groups=layout_config.logical_layer_groups,
                    head_dim=indexer_dim // compress_ratio,
                    **common,
                ),
                page_size,
                device,
            )
            self.indexer.storage_dtype = self.storage_dtype
            self.positions = None
            return
        self.indexer = QSAIndexerState(
            GQAAttentionConfig(logical_layer_groups=layout_config.logical_layer_groups, head_dim=indexer_dim, **common),
            page_size,
            device,
            torch.float8_e4m3fn if fp8 else layout_config.dtype,
        )
        layers = tuple(layer for group in layout_config.logical_layer_groups for layer in group)
        self.positions = GQAFullAttentionState(
            GQAAttentionConfig(logical_layer_groups=(layers,), head_dim=rotary_dim, **common), page_size, device
        )

    def get_state_type(self):
        suffix = "_fp8" if self.storage_dtype == torch.float8_e4m3fn else ""
        if self.compress_ratio > 1:
            suffix += f"_c{self.compress_ratio}"
        return "qwen4_" + super().get_state_type() + suffix

    def _side_states(self):
        return (self.indexer,) if self.positions is None else (self.indexer, self.positions)

    def allocate_storage(self, token_capacity):
        super().allocate_storage(token_capacity)
        for state in self._side_states():
            state.device = self.device
            state.allocate_storage(token_capacity)

    def get_token_bytes(self):
        return super().get_token_bytes() + sum(state.get_token_bytes() for state in self._side_states())

    def qsa_view(self, layer, compressor=None):
        if self.compress_ratio > 1:
            pooled = self.indexer.k_storage[self.indexer.get_storage_layer_index(layer)].view(-1, self.indexer_dim)
            return QSAExecutionState(None, pooled, None, None, compressor)
        raw, pooled = self.indexer.get_layer_view(layer)
        cos, sin = self.positions.get_layer_view(layer)
        return QSAExecutionState(raw[:, 0], pooled[:, 0], cos[:, 0], sin[:, 0])

    def store_positions(self, indices, cos, sin):
        if self.positions is None:
            return
        self.positions.k_storage[0, :, 0].index_copy_(0, indices, cos)
        self.positions.v_storage[0, :, 0].index_copy_(0, indices, sin)

    def _get_transfer_regions(self, page_tokens):
        regions = list(super()._get_transfer_regions(page_tokens))
        for prefix, state in (("indexer", self.indexer), ("positions", self.positions)):
            if state is None:
                continue
            offset = max(region.byte_end for region in regions)
            regions.extend(
                replace(
                    region,
                    field=prefix + "." + region.field,
                    byte_start=offset + region.byte_start,
                    byte_end=offset + region.byte_end,
                )
                for region in state._get_transfer_regions(page_tokens)
            )
        return tuple(regions)

    def _layouts(self, layout):
        for prefix, state in (("", super()), ("indexer.", self.indexer), ("positions.", self.positions)):
            if state is None:
                continue
            regions = tuple(
                replace(region, field=region.field[len(prefix) :])
                for region in layout.regions
                if (region.field.startswith(prefix) if prefix else "." not in region.field)
            )
            yield state, replace(layout, regions=regions)

    def write_to_cpu(self, token_indices, cpu_page_buffers, cpu_page_indices, layout, cpu_page_ready_flags=None):
        for state, part in self._layouts(layout):
            state.write_to_cpu(token_indices, cpu_page_buffers, cpu_page_indices, part, cpu_page_ready_flags)

    def read_from_cpu(self, cpu_page_buffers, cpu_page_indices, token_indices, layout):
        for state, part in self._layouts(layout):
            state.read_from_cpu(cpu_page_buffers, cpu_page_indices, token_indices, part)

    def write_to_pd_page(self, token_indices, pd_page_buffers, pd_page_index, layout):
        for state, part in self._layouts(layout):
            state.write_to_pd_page(token_indices, pd_page_buffers, pd_page_index, part)

    def read_from_pd_page(self, pd_page_buffers, token_indices, pd_page_index, layout):
        for state, part in self._layouts(layout):
            state.read_from_pd_page(pd_page_buffers, token_indices, pd_page_index, part)

    def copy_full_att_tokens(self, src_indices, dst_indices):
        super().copy_full_att_tokens(src_indices, dst_indices)
        for state in self._side_states():
            state.copy_full_att_tokens(src_indices, dst_indices)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        assert (
            self.storage_dtype == remote_dp_full_att_state.storage_dtype
            and self.compress_ratio == remote_dp_full_att_state.compress_ratio
        ), "QSA DP copy requires the same KV and indexer encoding"
        super().copy_tokens_from_dp(src_token_indices, dst_token_indices, remote_dp_full_att_state)
        for name in ("indexer", "positions"):
            if getattr(self, name) is None:
                continue
            getattr(self, name).copy_tokens_from_dp(
                src_token_indices, dst_token_indices, getattr(remote_dp_full_att_state, name)
            )
