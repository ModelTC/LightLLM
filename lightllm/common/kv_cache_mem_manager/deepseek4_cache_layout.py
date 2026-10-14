"""DeepSeek-V4 packed cache geometry shared by GPU pools, CPU checkpoints and PD."""

from dataclasses import dataclass
from typing import Sequence

import torch
import triton


# FlashMLA pages store 448 FP8 NoPE bytes + 64 BF16 RoPE values per row,
# followed by seven UE8M0 scales and one padding byte. Page strides align to a data row.
DSV4_MLA_DATA_BYTES_PER_TOKEN = 576
DSV4_MLA_SCALE_BYTES = 8
DSV4_SWA_PAGE_SIZE = 128
# Closes both compression groups; allocation/hash pages may contain multiple blocks.
DSV4_CACHE_BLOCK_SIZE = 256


def packed_page_nbytes(page_size: int, data_nbytes: int, scale_nbytes: int, align_nbytes: int = 1) -> int:
    return triton.cdiv(page_size * (data_nbytes + scale_nbytes), align_nbytes) * align_nbytes


@dataclass(frozen=True)
class _DeepseekV4CacheLayout:
    token_page_size: int
    history_block_num: int
    layer_num: int
    n_c4: int
    n_c128: int
    head_dim: int
    indexer_head_dim: int

    c4_offset: int
    c4_gpu_page_nbytes: int
    c4_gpu_pages_per_page: int
    c4_layer_nbytes: int
    c4_nbytes: int

    c4_indexer_offset: int
    c4_indexer_gpu_page_nbytes: int
    c4_indexer_layer_nbytes: int
    c4_indexer_nbytes: int

    c128_offset: int
    c128_row_nbytes: int
    c128_rows_per_page: int
    c128_layer_nbytes: int
    c128_nbytes: int

    swa_offset: int
    swa_gpu_page_nbytes: int
    swa_gpu_pages_per_page: int
    swa_layer_nbytes: int
    swa_nbytes: int

    c4_state_offset: int
    c4_state_row_nbytes: int
    c4_state_rows: int
    c4_state_nbytes: int

    c4_indexer_state_offset: int
    c4_indexer_state_row_nbytes: int
    c4_indexer_state_nbytes: int

    @classmethod
    def _history_layout(
        cls,
        compress_rates: Sequence[int],
        token_page_size: int,
        head_dim: int,
        indexer_head_dim: int,
    ):
        rates = tuple(int(rate) for rate in compress_rates)
        if token_page_size <= 0 or token_page_size % DSV4_CACHE_BLOCK_SIZE != 0:
            raise ValueError(
                f"DeepSeek-V4 cache page size must be a positive multiple of "
                f"{DSV4_CACHE_BLOCK_SIZE}, got {token_page_size}"
            )
        if head_dim != 512 or indexer_head_dim != 128:
            raise ValueError("DeepSeek-V4 packed cache requires head_dim=512 and indexer_head_dim=128")

        layer_num = len(rates)
        n_c4 = rates.count(4)
        n_c128 = rates.count(128)
        history_block_num = token_page_size // DSV4_CACHE_BLOCK_SIZE

        c4_gpu_page_nbytes = packed_page_nbytes(
            DSV4_CACHE_BLOCK_SIZE // 4,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
        )
        c4_gpu_pages_per_page = history_block_num
        c4_layer_nbytes = c4_gpu_pages_per_page * c4_gpu_page_nbytes
        c4_offset = 0
        c4_nbytes = n_c4 * c4_layer_nbytes

        c4_indexer_gpu_page_nbytes = packed_page_nbytes(
            DSV4_CACHE_BLOCK_SIZE // 4,
            indexer_head_dim,
            torch.float32.itemsize,
        )
        c4_indexer_layer_nbytes = c4_gpu_pages_per_page * c4_indexer_gpu_page_nbytes
        c4_indexer_offset = c4_offset + c4_nbytes
        c4_indexer_nbytes = n_c4 * c4_indexer_layer_nbytes

        c128_row_nbytes = DSV4_MLA_DATA_BYTES_PER_TOKEN + DSV4_MLA_SCALE_BYTES
        c128_rows_per_page = token_page_size // 128
        c128_layer_nbytes = c128_rows_per_page * c128_row_nbytes
        c128_offset = c4_indexer_offset + c4_indexer_nbytes
        c128_nbytes = n_c128 * c128_layer_nbytes

        return dict(
            token_page_size=token_page_size,
            history_block_num=history_block_num,
            layer_num=layer_num,
            n_c4=n_c4,
            n_c128=n_c128,
            head_dim=head_dim,
            indexer_head_dim=indexer_head_dim,
            c4_offset=c4_offset,
            c4_gpu_page_nbytes=c4_gpu_page_nbytes,
            c4_gpu_pages_per_page=c4_gpu_pages_per_page,
            c4_layer_nbytes=c4_layer_nbytes,
            c4_nbytes=c4_nbytes,
            c4_indexer_offset=c4_indexer_offset,
            c4_indexer_gpu_page_nbytes=c4_indexer_gpu_page_nbytes,
            c4_indexer_layer_nbytes=c4_indexer_layer_nbytes,
            c4_indexer_nbytes=c4_indexer_nbytes,
            c128_offset=c128_offset,
            c128_row_nbytes=c128_row_nbytes,
            c128_rows_per_page=c128_rows_per_page,
            c128_layer_nbytes=c128_layer_nbytes,
            c128_nbytes=c128_nbytes,
            swa_offset=c128_offset + c128_nbytes,
        )


@dataclass(frozen=True)
class DeepseekV4CpuCacheLayout(_DeepseekV4CacheLayout):
    """CPU checkpoint ABI: compressed history plus the final 256-token continuation."""

    page_nbytes: int

    @classmethod
    def load_from_args(cls):
        from lightllm.utils.config_utils import (
            get_config_json,
            get_layer_num,
            get_head_dim,
            get_deepseek_v4_compress_rates,
        )
        from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args

        args = get_env_start_args()
        config = get_config_json(args.model_dir)
        layer_num = get_layer_num(args.model_dir) + get_added_mtp_kv_layer_num()
        return cls.from_compress_rates(
            get_deepseek_v4_compress_rates(config, layer_num),
            token_page_size=args.cpu_cache_token_page_size,
            head_dim=get_head_dim(args.model_dir),
            indexer_head_dim=config["index_head_dim"],
        )

    @classmethod
    def from_compress_rates(
        cls,
        compress_rates: Sequence[int],
        token_page_size: int = 2048,
        head_dim: int = 512,
        indexer_head_dim: int = 128,
    ) -> "DeepseekV4CpuCacheLayout":
        history = cls._history_layout(compress_rates, token_page_size, head_dim, indexer_head_dim)
        swa_gpu_page_nbytes = packed_page_nbytes(
            DSV4_SWA_PAGE_SIZE,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
        )
        swa_gpu_pages_per_page = DSV4_CACHE_BLOCK_SIZE // DSV4_SWA_PAGE_SIZE
        swa_layer_nbytes = swa_gpu_pages_per_page * swa_gpu_page_nbytes
        swa_nbytes = history["layer_num"] * swa_layer_nbytes

        c4_state_rows = 4
        c4_state_row_nbytes = 4 * head_dim * torch.float32.itemsize
        c4_state_offset = history["swa_offset"] + swa_nbytes
        c4_state_nbytes = history["n_c4"] * c4_state_rows * c4_state_row_nbytes

        c4_indexer_state_row_nbytes = 4 * indexer_head_dim * torch.float32.itemsize
        c4_indexer_state_offset = c4_state_offset + c4_state_nbytes
        c4_indexer_state_nbytes = history["n_c4"] * c4_state_rows * c4_indexer_state_row_nbytes

        return cls(
            **history,
            swa_gpu_page_nbytes=swa_gpu_page_nbytes,
            swa_gpu_pages_per_page=swa_gpu_pages_per_page,
            swa_layer_nbytes=swa_layer_nbytes,
            swa_nbytes=swa_nbytes,
            c4_state_offset=c4_state_offset,
            c4_state_row_nbytes=c4_state_row_nbytes,
            c4_state_rows=c4_state_rows,
            c4_state_nbytes=c4_state_nbytes,
            c4_indexer_state_offset=c4_indexer_state_offset,
            c4_indexer_state_row_nbytes=c4_indexer_state_row_nbytes,
            c4_indexer_state_nbytes=c4_indexer_state_nbytes,
            page_nbytes=c4_indexer_state_offset + c4_indexer_state_nbytes,
        )


@dataclass(frozen=True)
class DeepseekV4PDCacheLayout(_DeepseekV4CacheLayout):
    """PD page ABI: compressed history plus request-tail continuation state."""

    c128_state_offset: int
    c128_state_row_nbytes: int
    c128_state_rows: int
    c128_state_layer_nbytes: int
    c128_state_nbytes: int
    page_nbytes: int

    @classmethod
    def from_compress_rates(
        cls,
        compress_rates: Sequence[int],
        token_page_size: int,
        head_dim: int = 512,
        indexer_head_dim: int = 128,
    ) -> "DeepseekV4PDCacheLayout":
        history = cls._history_layout(compress_rates, token_page_size, head_dim, indexer_head_dim)
        swa_gpu_page_nbytes = packed_page_nbytes(
            DSV4_SWA_PAGE_SIZE,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
        )
        swa_gpu_pages_per_page = 4
        swa_layer_nbytes = swa_gpu_pages_per_page * swa_gpu_page_nbytes
        swa_nbytes = history["layer_num"] * swa_layer_nbytes

        # Four rows at the aligned checkpoint, plus up to seven live-tail rows.
        c4_state_rows = 4 + 2 * 4 - 1
        c4_state_row_nbytes = 4 * head_dim * torch.float32.itemsize
        c4_state_offset = history["swa_offset"] + swa_nbytes
        c4_state_nbytes = history["n_c4"] * c4_state_rows * c4_state_row_nbytes

        c4_indexer_state_row_nbytes = 4 * indexer_head_dim * torch.float32.itemsize
        c4_indexer_state_offset = c4_state_offset + c4_state_nbytes
        c4_indexer_state_nbytes = history["n_c4"] * c4_state_rows * c4_indexer_state_row_nbytes

        c128_state_rows = 128 - 1
        c128_state_row_nbytes = 2 * head_dim * torch.float32.itemsize
        c128_state_layer_nbytes = c128_state_rows * c128_state_row_nbytes
        c128_state_offset = c4_indexer_state_offset + c4_indexer_state_nbytes
        c128_state_nbytes = history["n_c128"] * c128_state_layer_nbytes

        return cls(
            **history,
            swa_gpu_page_nbytes=swa_gpu_page_nbytes,
            swa_gpu_pages_per_page=swa_gpu_pages_per_page,
            swa_layer_nbytes=swa_layer_nbytes,
            swa_nbytes=swa_nbytes,
            c4_state_offset=c4_state_offset,
            c4_state_row_nbytes=c4_state_row_nbytes,
            c4_state_rows=c4_state_rows,
            c4_state_nbytes=c4_state_nbytes,
            c4_indexer_state_offset=c4_indexer_state_offset,
            c4_indexer_state_row_nbytes=c4_indexer_state_row_nbytes,
            c4_indexer_state_nbytes=c4_indexer_state_nbytes,
            c128_state_offset=c128_state_offset,
            c128_state_row_nbytes=c128_state_row_nbytes,
            c128_state_rows=c128_state_rows,
            c128_state_layer_nbytes=c128_state_layer_nbytes,
            c128_state_nbytes=c128_state_nbytes,
            page_nbytes=c128_state_offset + c128_state_nbytes,
        )
