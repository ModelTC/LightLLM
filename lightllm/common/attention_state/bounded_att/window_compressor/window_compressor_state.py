from functools import partial, lru_cache
import torch
from lightllm.utils.envs_utils import get_env_start_args
from ...utils import deepcopy_result
from ...types import TransferRegion
from ..formats import WindowCompressorStateConfig, compressor_spec
from ..state import WindowBoundedState
from ..window_pages_pool import WindowPagesPool


class WindowCompressorState(WindowBoundedState):
    def __init__(
        self,
        config: WindowCompressorStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ):
        from lightllm.common.attention_compute.bounded import CompressorRuntime

        spec = compressor_spec(config, request_capacity, mtp_step)
        pool = (
            partial(
                WindowPagesPool,
                snapshot_shape=spec.snapshot_fields[0].shape,
                dtype=spec.snapshot_fields[0].dtype,
            )
            if spec.snapshot_fields
            else None
        )
        super().__init__(
            config=config,
            spec=spec,
            runtime=CompressorRuntime(config),
            request_capacity=request_capacity,
            mtp_step=mtp_step,
            device=device,
            page_pool_factory=pool,
        )

    @property
    def buffer(self):
        return self.resource.buffers.get("buffer")

    @buffer.setter
    def buffer(self, value):
        self.resource.buffers["buffer"] = value

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """校验缓存页对齐产出步长；无重叠历史的窗口不占 CPU 页，也不创建大小页池。"""
        args = get_env_start_args()
        assert (
            args.linear_att_hash_page_size % self.config.compress_stride == 0
        ), "state cache pages must align to the compressor stride"
        if args.cpu_cache_token_page_size is not None:
            assert (
                args.cpu_cache_token_page_size % self.config.compress_stride == 0
            ), "CPU cache pages must align to the compressor stride"
        return self.resource.spec.cpu_regions
