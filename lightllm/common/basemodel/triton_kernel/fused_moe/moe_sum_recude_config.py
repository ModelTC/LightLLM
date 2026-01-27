import os
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MoeSumReduceKernelConfig(KernelConfigs):
    kernel_name: str = "moe_sum_reduce_kernel"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        topk_num: int,
        hidden_dim: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "topk_num": topk_num,
            "hidden_dim": hidden_dim,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            config = {
                "BLOCK_M": 1,
                "BLOCK_DIM": 128,
                "NUM_STAGE": 1,
                "num_warps": 2,
            }

        return config

    @classmethod
    def save_config(
        cls,
        topk_num: int,
        hidden_dim: int,
        out_dtype: str,
        config_json: dict,
    ):
        key_params = {
            "topk_num": topk_num,
            "hidden_dim": hidden_dim,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)
