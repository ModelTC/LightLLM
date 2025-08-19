import os
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MoeSiluAndMulKernelConfig(KernelConfigs):
    kernel_name: str = "moe_silu_and_mul_kernel"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        N: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "N": N,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            if M < 256:
                config = {"BLOCK_M": 1, "BLOCK_N": 128, "num_warps": 1, "NUM_STAGES": 1}
            else:
                config = {"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "NUM_STAGES": 5}

        return config

    @classmethod
    def save_config(
        cls,
        N: int,
        out_dtype: str,
        config_json: dict,
    ):
        key_params = {
            "N": N,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)
