from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class MlaDecodeAttentionKernelConfig(KernelConfigs):
    kernel_name: str = "mla_decode_attentnion"

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        batch_size: int,
        avg_seq_len_in_batch: int,
        q_head_num: int,
        q_head_dim: int,
        q_rope_dim: int,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "q_head_num": q_head_num,
            "q_head_dim": q_head_dim,
            "q_rope_dim": q_rope_dim,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            # two search dim, first is avg_seq_len_in_batch, second is batch_size
            batch_size_config: dict = finded_config[
                min(finded_config.keys(), key=lambda x: abs(int(x) - avg_seq_len_in_batch))
            ]
            config = batch_size_config[min(batch_size_config.keys(), key=lambda x: abs(int(x) - batch_size))]

            return config
        else:
            config = {
                "BLOCK_N": 16,
                "BLOCK_Q_HEAD": 16,
                "stage1_num_warps": 4,
                "stage1_num_stages": 2,
                "stage2_num_warps": 4,
                "stage2_num_stages": 2,
            }
        return config

    @classmethod
    def save_config(
        cls, q_head_num: int, q_head_dim: int, q_rope_dim: int, out_dtype: str, config_json: Dict[int, Dict[int, Dict]]
    ):

        key_params = {
            "q_head_num": q_head_num,
            "q_head_dim": q_head_dim,
            "q_rope_dim": q_rope_dim,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)
