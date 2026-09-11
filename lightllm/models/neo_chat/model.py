from __future__ import annotations

import json
import os

import torch

from lightllm.common.build_utils import repair_config
from lightllm.models.neo_chat.infer_struct import NeoChatInferStateInfo
from lightllm.models.neo_chat.layer_infer.transformer_layer_infer import (
    NeoChatTransformerLayerInfer,
)
from lightllm.models.neo_chat.layer_weights.pre_and_post_layer_weight import (
    NeoChatPreAndPostLayerWeight,
)
from lightllm.models.neo_chat.layer_weights.transformer_layer_weight import (
    NeoChatTransformerLayerWeight,
)
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import (
    LlamaMultimodalPreLayerInfer,
)
from lightllm.models.registry import ModelRegistry, llm_model_type_is


def build_neo_rope_tables(
    *,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_theta_hw: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Neo temporal and spatial RoPE tables without global model changes."""

    if head_dim % 8 != 0:
        raise ValueError(f"Neo head_dim must be divisible by 8, got {head_dim}")
    if max_seq_len <= 0:
        raise ValueError("Neo max_seq_len must be positive")

    def table(rotary_dim: int, theta: float):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        return (
            torch.cos(freqs).to(device=device, dtype=dtype),
            torch.sin(freqs).to(device=device, dtype=dtype),
        )

    temporal_cos, temporal_sin = table(head_dim // 2, rope_theta)
    spatial_cos, spatial_sin = table(head_dim // 4, rope_theta_hw)
    return temporal_cos, temporal_sin, spatial_cos, spatial_sin


@ModelRegistry(
    "neo_chat",
    is_multimodal=True,
    condition=llm_model_type_is("qwen3"),
)
class NeoTpPartModel(Qwen3TpPartModel):
    """Neo dense understanding tower backed by the normal LightLLM runtime."""

    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = NeoChatTransformerLayerInfer
    pre_and_post_weight_class = NeoChatPreAndPostLayerWeight
    transformer_weight_class = NeoChatTransformerLayerWeight
    infer_state_class = NeoChatInferStateInfo

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as config_file:
            outer_config = json.load(config_file)
        if not isinstance(outer_config.get("llm_config"), dict):
            raise ValueError("Neo config.json must contain an llm_config object")
        self.outer_config = outer_config
        self.config = dict(outer_config["llm_config"])
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        self.config.setdefault(
            "num_key_value_heads",
            self.config["num_attention_heads"],
        )
        self.config.setdefault(
            "head_dim",
            self.config["hidden_size"] // self.config["num_attention_heads"],
        )
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def _verify_params(self):
        super()._verify_params()
        if self.config["head_dim"] % 8:
            raise ValueError("Neo head_dim must be divisible by 8")
        if self.args.llm_kv_type != "None":
            raise ValueError("Neo mixed-mask prefill currently requires unquantized KV")
        if self.args.enable_dp_prefill_balance:
            raise ValueError("Neo does not currently support DP prefill balance")
        if self.args.enable_tpsp_mix_mode:
            raise ValueError(
                "Neo mixed-mask prefill does not currently support TPSP mix mode"
            )

    def _init_custom(self):
        # Neo uses three independent RoPE axes.  Keep these caches local to
        # this model instead of altering Llama/Qwen behavior globally.
        max_position = int(self.config.get("max_position_embeddings", 16_384))
        table_len = max(self.max_seq_length, max_position + 1024 * 128)
        (
            self._cos_cached,
            self._sin_cached,
            self._hw_cos_cached,
            self._hw_sin_cached,
        ) = build_neo_rope_tables(
            head_dim=int(self.config["head_dim"]),
            max_seq_len=table_len,
            rope_theta=float(self.config.get("rope_theta", 10_000.0)),
            rope_theta_hw=float(self.config.get("rope_theta_hw", 10_000.0)),
            dtype=self.data_type,
            device="cuda",
        )
