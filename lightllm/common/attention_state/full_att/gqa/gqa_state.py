import torch
from .formats import GQAAttentionConfig, plain_spec
from .state import GQAStorageState


class GQAFullAttentionState(GQAStorageState):
    def __init__(self, layout_config: GQAAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"):
        from lightllm.common.attention_compute.gqa import PlainKVWriter

        super().__init__(layout_config, plain_spec(layout_config), PlainKVWriter(layout_config), page_size, device)
