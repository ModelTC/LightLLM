import torch
from .formats import StaticFP8AttentionConfig, static_fp8_spec
from .state import GQAStorageState


class StaticFP8FullAttentionState(GQAStorageState):
    def __init__(
        self, layout_config: StaticFP8AttentionConfig, page_size: int = 1, device: str | torch.device = "cuda"
    ):
        from lightllm.common.attention_compute.gqa import StaticFP8KVWriter

        super().__init__(
            layout_config, static_fp8_spec(layout_config), StaticFP8KVWriter(layout_config), page_size, device
        )

    def bind_calibration(self, calibration, storage_layer_start: int) -> None:
        self.compute.bind_calibration(calibration, storage_layer_start)

    @property
    def scales(self):
        return self.compute.scales

    @property
    def q_scales(self):
        return self.compute.q_scales

    @property
    def cpu_scales(self):
        return self.compute.cpu_scales
