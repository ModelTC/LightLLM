"""Qwen1 configuration fields, including its pre-Qwen2 attention controls.

Defaults follow Qwen/Qwen-7B@ef3c5c9c57b252f3149c1408daf4d649ec8b6c85.
This local Config uses HF's attribute storage/serialization and loads no model code.
"""

from transformers import PretrainedConfig


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]
    _defaults = {
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 22016,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 8192,
        "kv_channels": 128,
        "emb_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "layer_norm_epsilon": 1e-6,
        "initializer_range": 0.02,
        "rotary_pct": 1.0,
        "rotary_emb_base": 10000,
        "scale_attn_weights": True,
        "use_cache": True,
        "bf16": False,
        "fp16": False,
        "fp32": False,
        "use_dynamic_ntk": True,
        "use_logn_attn": True,
        "use_flash_attn": "auto",
        "no_bias": True,
        "tie_word_embeddings": False,
        "use_cache_quantization": False,
        "use_cache_kernel": False,
        "softmax_in_fp32": False,
    }

    def __init__(self, **kwargs):
        super().__init__(**{**self._defaults, **kwargs})
