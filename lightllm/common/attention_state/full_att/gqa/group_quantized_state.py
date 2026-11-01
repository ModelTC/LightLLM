import torch
from .formats import GroupQuantizedAttentionConfig, group_quantized_spec
from .state import GQAStorageState


class GroupQuantizedFullAttentionState(GQAStorageState):
    def __init__(
        self, layout_config: GroupQuantizedAttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        from lightllm.common.attention_compute.gqa import GroupQuantizedKVWriter

        super().__init__(
            layout_config, group_quantized_spec(layout_config), GroupQuantizedKVWriter(layout_config), page_size, device
        )

    @property
    def k_scales(self):
        return self.storage.buffers.get("k_scale")

    @k_scales.setter
    def k_scales(self, buffer):
        self.storage.buffers["k_scale"] = buffer

    @property
    def v_scales(self):
        return self.storage.buffers.get("v_scale")

    @v_scales.setter
    def v_scales(self, buffer):
        self.storage.buffers["v_scale"] = buffer
