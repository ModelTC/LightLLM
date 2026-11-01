"""序列压缩 attention 历史 KV 的独立创建入口。"""

import torch

from .compressed_att_state import CompressedAttentionConfig, CompressedAttentionState


def create_compressed_att_state(
    config: CompressedAttentionConfig,
    page_size: int = 256,
    device: str | torch.device = "cuda",
) -> CompressedAttentionState:
    """根据压缩比例创建历史 KV 定义；不分配存储，也不创建 SWA/compressor 运行态。"""
    return CompressedAttentionState(layout_config=config, page_size=page_size, device=device)
