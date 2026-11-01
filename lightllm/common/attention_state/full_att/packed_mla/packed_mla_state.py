import torch
from .formats import PackedMLAAttentionConfig, PackedMLALogicalConfig, PackedMLALayerView, packed_mla_spec
from ..state import KVWritableState, SingleTensorAttentionState


class PackedMLAFullAttentionState(KVWritableState, SingleTensorAttentionState):
    def __init__(
        self, layout_config: PackedMLAAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        from lightllm.common.attention_compute.mla import PackedMLAKVWriter

        super().__init__(
            config=layout_config,
            spec=packed_mla_spec(layout_config),
            compute=PackedMLAKVWriter(layout_config),
            page_size=page_size,
            device=device,
        )
