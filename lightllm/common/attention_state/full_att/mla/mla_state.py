import torch
from .formats import MLAConfig, mla_spec
from ..state import KVWritableState, SingleTensorAttentionState


class MLAFullAttentionState(KVWritableState, SingleTensorAttentionState):
    def __init__(self, layout_config: MLAConfig, page_size: int = 1, device: str | torch.device = "cuda"):
        from lightllm.common.attention_compute.mla import MLAKVWriter

        super().__init__(
            config=layout_config,
            spec=mla_spec(layout_config),
            compute=MLAKVWriter(layout_config),
            page_size=page_size,
            device=device,
        )
