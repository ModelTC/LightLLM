import os
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class DeepseekV3RotaryKernelConfig(KernelConfigs):
    kernel_name: str = "deepseek_v3_rotary_emb_kernel"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        Q_HEAD_NUM: int,
        K_HEAD_NUM: int,
        HEAD_DIM: int,
        dtype: str,
    ) -> dict:
        key_params = {
            "Q_HEAD_NUM": Q_HEAD_NUM,
            "K_HEAD_NUM": K_HEAD_NUM,
            "HEAD_DIM": HEAD_DIM,
            "dtype": str(dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            if M <= 256:
                config = {"BLOCK_SEQ": 1, "NUM_STAGE": 1, "num_warps": 1, "num_stages": 1, "HEAD_PARALLEL_NUM": 1}
            else:
                config = {"BLOCK_SEQ": 16, "NUM_STAGE": 1, "num_warps": 1, "num_stages": 1, "HEAD_PARALLEL_NUM": 1}

        return config

    @classmethod
    def save_config(
        cls,
        Q_HEAD_NUM: int,
        K_HEAD_NUM: int,
        HEAD_DIM: int,
        dtype: str,
        config_json: dict,
    ):
        key_params = {
            "Q_HEAD_NUM": Q_HEAD_NUM,
            "K_HEAD_NUM": K_HEAD_NUM,
            "HEAD_DIM": HEAD_DIM,
            "dtype": str(dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)
