from functools import partial
import torch
from ..formats import FP8E8M0SlidingWindowConfig, sliding_window_spec
from ..state import WindowBoundedState
from ..window_pages_pool import WindowPagesPool


class FP8E8M0SlidingWindowState(WindowBoundedState):
    def __init__(
        self,
        config: FP8E8M0SlidingWindowConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ):
        from lightllm.common.attention_compute.bounded import SlidingWindowRuntime

        spec = sliding_window_spec(config, request_capacity, mtp_step)
        snapshot = spec.snapshot_fields[0]
        pool = partial(
            WindowPagesPool,
            snapshot_shape=snapshot.shape,
            dtype=snapshot.dtype,
        )
        super().__init__(
            config=config,
            spec=spec,
            runtime=SlidingWindowRuntime(config),
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=pool,
        )
        self.swa_capacity_tokens = self.swa_buffer.shape[2]
        self.HOLD_REQUEST_ID = request_capacity - 1
        self.HOLD_SWA_SLOT = self.HOLD_REQUEST_ID * self.swa_capacity_tokens

    @property
    def swa_buffer(self):
        return self.resource.buffers.get("swa_buffer")

    @swa_buffer.setter
    def swa_buffer(self, value):
        self.resource.buffers["swa_buffer"] = value
