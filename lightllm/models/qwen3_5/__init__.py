"""
Qwen3.5 Multimodal Model Module (Dense Variant)

Provides Qwen3.5 dense multimodal model with hybrid attention and vision-language support.
For MoE variant, see qwen3_5_moe module.
"""

from .model import (
    Qwen3_5TpPartModel,
    QWen3_5Tokenizer,
)

__all__ = [
    "Qwen3_5TpPartModel",
    "QWen3_5Tokenizer",
]
