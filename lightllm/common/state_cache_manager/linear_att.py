from .base import StateCacheManager
from .layer_cache import LayerCache
from .linear_att_config import LinearAttCacheConfig


class LinearAttCacheManager(StateCacheManager):
    """CPU pinned conv/SSM checkpoint，两个 buffer 均保持 size-first 布局。"""

    def __init__(
        self,
        size: int,
        linear_config: LinearAttCacheConfig,
        keep_num: int = 0,  # 用于记录需要保留的缓存数量，用于支持含有 linear_att 的如qwen3.5 模型的cpu cache的碎页处理。
    ):
        super().__init__(size, keep_num)
        self.linear_config = linear_config
        # init the layer cache
        self.conv_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_conv_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
            size_first=True,
        )
        self.ssm_state_cache = LayerCache(
            size=self.size,
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cpu",
            size_first=True,
        )
        return

    def get_state_cache(self, buffer_idx: int):
        return self.conv_state_cache.buffer[buffer_idx, ...], self.ssm_state_cache.buffer[buffer_idx, ...]

    def clear_to_init_state(self):
        self.conv_state_cache.buffer.zero_()
        self.ssm_state_cache.buffer.zero_()
        super().clear_to_init_state()
        return
