import torch
from .formats import FP8E8M0PackedMLAConfig, fp8_e8m0_spec
from ..state import KVWritableState, SingleTensorAttentionState


class FP8E8M0PackedMLAState(KVWritableState, SingleTensorAttentionState):
    def __init__(self, layout_config: FP8E8M0PackedMLAConfig, page_size: int = 1, device: str | torch.device = "cuda"):
        from lightllm.common.attention_compute.mla import FP8E8M0KVWriter

        super().__init__(
            config=layout_config,
            spec=fp8_e8m0_spec(layout_config),
            compute=FP8E8M0KVWriter(layout_config),
            page_size=page_size,
            device=device,
        )
