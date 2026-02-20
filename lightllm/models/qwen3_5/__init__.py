"""
Qwen3.5 Multimodal Model Module

Provides Qwen3.5 multimodal models with hybrid attention and vision-language support.
"""

from .model import (
    Qwen3_5TpPartModel,
    Qwen3_5MOETpPartModel,
    QWen3_5Tokenizer,
)

__all__ = [
    "Qwen3_5TpPartModel",
    "Qwen3_5MOETpPartModel",
    "QWen3_5Tokenizer",
]
