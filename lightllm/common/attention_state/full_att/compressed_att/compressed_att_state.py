import torch
from .formats import CompressedAttentionConfig, CompressedAttentionLayerView, compressed_spec
from ..state import SingleTensorAttentionState
from ..token_transfer import BlockTokenTransfer


class CompressedAttentionState(SingleTensorAttentionState):
    def __init__(
        self, layout_config: CompressedAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        from lightllm.common.attention_compute.mla import CompressedAttentionCompute

        super().__init__(
            config=layout_config,
            spec=compressed_spec(layout_config),
            compute=CompressedAttentionCompute(layout_config),
            page_size=page_size,
            device=device,
        )
        self.transfer = BlockTokenTransfer(self.resource)
