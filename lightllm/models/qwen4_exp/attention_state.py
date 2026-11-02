from dataclasses import replace

import torch

from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAFullAttentionState, GQAAttentionConfig
from .state import QSAExecutionState


class Qwen4ExpAttentionState(GQAFullAttentionState):
    def __init__(self, layout_config, indexer_dim, rotary_dim, page_size=1, device="meta"):
        super().__init__(layout_config, page_size, device)
        common = dict(
            global_head_num=1,
            dtype=layout_config.dtype,
            tp_rank=layout_config.tp_rank,
            tp_world_size=layout_config.tp_world_size,
            replicated=True,
        )
        self.indexer = GQAFullAttentionState(
            GQAAttentionConfig(logical_layer_groups=layout_config.logical_layer_groups, head_dim=indexer_dim, **common),
            page_size,
            device,
        )
        layers = tuple(layer for group in layout_config.logical_layer_groups for layer in group)
        self.positions = GQAFullAttentionState(
            GQAAttentionConfig(logical_layer_groups=(layers,), head_dim=rotary_dim, **common), page_size, device
        )

    def get_state_type(self):
        return "qwen4_" + super().get_state_type()

    def allocate_storage(self, token_capacity):
        super().allocate_storage(token_capacity)
        for state in (self.indexer, self.positions):
            state.device = self.device
            state.allocate_storage(token_capacity)

    def get_token_bytes(self):
        return super().get_token_bytes() + self.indexer.get_token_bytes() + self.positions.get_token_bytes()

    def qsa_view(self, layer):
        raw, pooled = self.indexer.get_layer_view(layer)
        cos, sin = self.positions.get_layer_view(layer)
        return QSAExecutionState(raw[:, 0], pooled[:, 0], cos[:, 0], sin[:, 0])

    def store_positions(self, indices, cos, sin):
        self.positions.k_storage[0, :, 0].index_copy_(0, indices, cos)
        self.positions.v_storage[0, :, 0].index_copy_(0, indices, sin)

    def _get_transfer_regions(self, page_tokens):
        regions = list(super()._get_transfer_regions(page_tokens))
        for prefix, state in (("indexer", self.indexer), ("positions", self.positions)):
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
        for state in (self.indexer, self.positions):
            state.copy_full_att_tokens(src_indices, dst_indices)

    def copy_tokens_from_dp(self, src_token_indices, dst_token_indices, remote_dp_full_att_state):
        super().copy_tokens_from_dp(src_token_indices, dst_token_indices, remote_dp_full_att_state)
        for name in ("indexer", "positions"):
            getattr(self, name).copy_tokens_from_dp(
                src_token_indices, dst_token_indices, getattr(remote_dp_full_att_state, name)
            )
