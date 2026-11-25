from dataclasses import dataclass, field
from typing import ClassVar

from lightllm.common.attention_state.bounded_att import WindowCompressorStateConfig
from lightllm.common.attention_state.full_att.compressed_att.compressed_att_state import CompressedAttentionConfig
from lightllm.models.deepseek_v4.config import DSV4AttentionConfig


@dataclass(frozen=True)
class DSV41AttentionConfig(DSV4AttentionConfig):
    supported_compress_ratios: ClassVar = (0, 1, 2)
    kv_source_layer_ids: tuple[int, ...] = field(kw_only=True)

    @classmethod
    def from_model_config(cls, model_config, model_logical_layer_ids, tp_rank=0, tp_world_size=1):
        return super().from_model_config(
            model_config,
            model_logical_layer_ids,
            tp_rank,
            tp_world_size,
            kv_source_layer_ids=tuple(model_config["kv_source_layer_ids"]),
        )

    def create_compressed_config(self, compress_ratio):
        # Consumers of the same ratio share the preceding source layer's KV and indexer pages.
        groups = []
        for layer, ratio in zip(self.logical_layer_ids, self.compress_ratios):
            if ratio == compress_ratio:
                if layer in self.kv_source_layer_ids:
                    groups.append([])
                groups[-1].append(layer)
        return CompressedAttentionConfig(
            logical_layer_groups=tuple(tuple(group) for group in groups),
            compress_ratio=compress_ratio,
            kv_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            block_token_size=64 * compress_ratio,
            indexer_dim=self.indexer_head_dim,
            tp_rank=self.tp_rank,
            tp_world_size=self.tp_world_size,
        )

    def compressor_configs(self):
        # Only C2 sources retain a pending pair. C1 writes its latent directly.
        layers = tuple(
            layer
            for layer, ratio in zip(self.logical_layer_ids, self.compress_ratios)
            if ratio == 2 and layer in self.kv_source_layer_ids
        )
        if layers:
            yield WindowCompressorStateConfig(
                logical_layer_ids=layers,
                window_size=2,
                compress_stride=2,
                candidate_dim=self.head_dim,
                target="kv",
                tp_rank=self.tp_rank,
                tp_world_size=self.tp_world_size,
            )
