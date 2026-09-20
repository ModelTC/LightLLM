import os

import torch
from dataclasses import dataclass
from typing import List, Optional, Sequence
from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from .allocator import KvCacheAllocator
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# fp8_ds_mla packed-latent byte layout (ABI shared with the flash_mla extra-cache fork and
# sglang/vllm): 448B NoPE fp8 + 64*2B RoPE bf16 + 7B ue8m0 scale + 1B pad = 584B per token,
# stored in packed GPU pages whose tail carries the per-token scale bytes.
DSV4_MLA_NOPE_DIM = 448  # 448B
DSV4_MLA_ROPE_DIM = 64  # 64 dim
DSV4_MLA_HEAD_DIM = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM  # 512
DSV4_MLA_QUANT_GROUP_SIZE = 64  # 64
DSV4_MLA_SCALE_BYTES = DSV4_MLA_NOPE_DIM // DSV4_MLA_QUANT_GROUP_SIZE + 1  # 8 (7 ue8m0 + 1 pad)
DSV4_MLA_BYTES_PER_TOKEN = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM * 2 + DSV4_MLA_SCALE_BYTES  # 584
DSV4_MLA_DATA_BYTES_PER_TOKEN = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM * 2  # 576
DSV4_MLA_PAGE_ALIGN_BYTES = DSV4_MLA_DATA_BYTES_PER_TOKEN  # 576
DSV4_INDEXER_HEAD_DIM = 128  # 128
DSV4_INDEXER_SCALE_BYTES = 4  # 4B fp32 scale
DSV4_INDEXER_BYTES_PER_TOKEN = DSV4_INDEXER_HEAD_DIM + DSV4_INDEXER_SCALE_BYTES  # 132
DSV4_FP8_E4M3_MAX = 448.0  # 448.0
DSV4_FP8_AMAX_MIN = 1e-4  # 1e-4
DSV4_SWA_PAGE_SIZE = 128  # 128 slots/page
DSV4_C4_PAGE_SIZE = 64  # 64 slots/page
DSV4_C128_PAGE_SIZE = 2  # 2 slots/page
DSV4_PROMPT_CACHE_PAGE_SIZE = DSV4_C4_PAGE_SIZE * 4  # 256 (= c4 ratio)
DSV4_CPU_CACHE_TOKEN_PAGE_SIZE = 2048
# compressor state ring: c4 overlap 对的基础窗口为每页 2 个分组槽 × ratio 4 行；MTP
# 追加候选槽，避免 rejected draft 覆盖仍存活的基础窗口。c128 同样追加候选槽，再对齐到 ratio 4。
DSV4_C4_STATE_RING = 8  # 8 rows/page before MTP padding
DSV4_C128_STATE_RING = 128  # 128 rows/request before MTP padding
# swa 池占 full token 空间的比例(sglang DSV4 默认 swa_full_tokens_ratio=0.1 同值)。
# 瞬时借页/驱逐走 swa 压力阀;池子大小仅按 ratio 切分,不再叠加结构性余量。
DSV4_SWA_FULL_TOKENS_RATIO = float(os.getenv("DSV4_SWA_FULL_TOKENS_RATIO", "0.1"))


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _aligned_gpu_page_nbytes(page_size: int, data_nbytes: int, scale_nbytes: int, align_nbytes: int = 1) -> int:
    return _ceil_div(page_size * (data_nbytes + scale_nbytes), align_nbytes) * align_nbytes


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
        if token_page_size <= 0 or token_page_size % DSV4_PROMPT_CACHE_PAGE_SIZE != 0:
            raise ValueError(
                f"DeepSeek-V4 cache page size must be a positive multiple of "
                f"{DSV4_PROMPT_CACHE_PAGE_SIZE}, got {token_page_size}"
            )
        if head_dim != DSV4_MLA_HEAD_DIM:
            raise ValueError(f"DeepSeek-V4 cache expects head_dim={DSV4_MLA_HEAD_DIM}, got {head_dim}")
        if indexer_head_dim != DSV4_INDEXER_HEAD_DIM:
            raise ValueError(
                f"DeepSeek-V4 cache expects indexer_head_dim={DSV4_INDEXER_HEAD_DIM}, got {indexer_head_dim}"
            )

        layer_num = len(rates)
        n_c4 = rates.count(4)
        n_c128 = rates.count(128)
        history_block_num = token_page_size // DSV4_PROMPT_CACHE_PAGE_SIZE

        c4_gpu_page_nbytes = _aligned_gpu_page_nbytes(
            DSV4_C4_PAGE_SIZE,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_PAGE_ALIGN_BYTES,
        )
        c4_gpu_pages_per_page = history_block_num
        c4_layer_nbytes = c4_gpu_pages_per_page * c4_gpu_page_nbytes
        c4_offset = 0
        c4_nbytes = n_c4 * c4_layer_nbytes

        c4_indexer_gpu_page_nbytes = _aligned_gpu_page_nbytes(
            DSV4_C4_PAGE_SIZE,
            indexer_head_dim,
            DSV4_INDEXER_SCALE_BYTES,
        )
        c4_indexer_layer_nbytes = c4_gpu_pages_per_page * c4_indexer_gpu_page_nbytes
        c4_indexer_offset = c4_offset + c4_nbytes
        c4_indexer_nbytes = n_c4 * c4_indexer_layer_nbytes

        c128_row_nbytes = DSV4_MLA_BYTES_PER_TOKEN
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
        from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num

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
        token_page_size: int = DSV4_CPU_CACHE_TOKEN_PAGE_SIZE,
        head_dim: int = DSV4_MLA_HEAD_DIM,
        indexer_head_dim: int = DSV4_INDEXER_HEAD_DIM,
    ) -> "DeepseekV4CpuCacheLayout":
        history = cls._history_layout(compress_rates, token_page_size, head_dim, indexer_head_dim)
        swa_gpu_page_nbytes = _aligned_gpu_page_nbytes(
            DSV4_SWA_PAGE_SIZE,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_PAGE_ALIGN_BYTES,
        )
        swa_gpu_pages_per_page = DSV4_PROMPT_CACHE_PAGE_SIZE // DSV4_SWA_PAGE_SIZE
        swa_layer_nbytes = swa_gpu_pages_per_page * swa_gpu_page_nbytes
        swa_nbytes = history["layer_num"] * swa_layer_nbytes

        c4_state_rows = 4
        c4_state_row_nbytes = 4 * head_dim * torch._utils._element_size(torch.float32)
        c4_state_offset = history["swa_offset"] + swa_nbytes
        c4_state_nbytes = history["n_c4"] * c4_state_rows * c4_state_row_nbytes

        c4_indexer_state_row_nbytes = 4 * indexer_head_dim * torch._utils._element_size(torch.float32)
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
        head_dim: int = DSV4_MLA_HEAD_DIM,
        indexer_head_dim: int = DSV4_INDEXER_HEAD_DIM,
    ) -> "DeepseekV4PDCacheLayout":
        history = cls._history_layout(compress_rates, token_page_size, head_dim, indexer_head_dim)
        swa_gpu_page_nbytes = _aligned_gpu_page_nbytes(
            DSV4_SWA_PAGE_SIZE,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_PAGE_ALIGN_BYTES,
        )
        swa_gpu_pages_per_page = 4
        swa_layer_nbytes = swa_gpu_pages_per_page * swa_gpu_page_nbytes
        swa_nbytes = history["layer_num"] * swa_layer_nbytes

        # Four rows at the aligned checkpoint, plus up to seven live-tail rows.
        c4_state_rows = 4 + DSV4_C4_STATE_RING - 1
        c4_state_row_nbytes = 4 * head_dim * torch._utils._element_size(torch.float32)
        c4_state_offset = history["swa_offset"] + swa_nbytes
        c4_state_nbytes = history["n_c4"] * c4_state_rows * c4_state_row_nbytes

        c4_indexer_state_row_nbytes = 4 * indexer_head_dim * torch._utils._element_size(torch.float32)
        c4_indexer_state_offset = c4_state_offset + c4_state_nbytes
        c4_indexer_state_nbytes = history["n_c4"] * c4_state_rows * c4_indexer_state_row_nbytes

        c128_state_rows = DSV4_C128_STATE_RING - 1
        c128_state_row_nbytes = 2 * head_dim * torch._utils._element_size(torch.float32)
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


@dataclass(frozen=True)
class DeepseekV4CpuCacheLoadPlan:
    loaded_start: int
    loaded_end: int
    mem_indexes: torch.Tensor
    history_full_slots: torch.Tensor
    history_c4_slots: Optional[torch.Tensor]
    history_c128_slots: Optional[torch.Tensor]
    resume_swa_slots: torch.Tensor


class PackedPagePool:
    """fp8_ds_mla 风格的 packed page 存储: 每页前段连续放 token 的 data 字节，页尾放 per-token scale 字节。

    寻址是纯 token 槽位 (page = slot // page_size)，page 只是 scale-tail/对齐的物理打包技巧，
    不存在页粒度的分配。``write``/``read`` 是 torch 参考实现(单测 oracle)；生产写入走
    triton packed writer(destindex_copy_kv_flashmla_dsv4 等)，kernel 直接消费 ``buffer``。
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        data_bytes: int,
        scale_bytes: int,
        align_bytes: int = 1,
        device: str = "cuda",
    ):
        self.size = size
        self.page_size = page_size
        self.layer_num = layer_num
        self.data_bytes_per_token = data_bytes
        self.scale_bytes_per_token = scale_bytes
        self.bytes_per_token = data_bytes + scale_bytes
        self.num_pages = _ceil_div(size + 1, page_size)
        self.bytes_per_page = _ceil_div(page_size * self.bytes_per_token, align_bytes) * align_bytes
        self.scale_offset_in_page = page_size * data_bytes
        self.buffer = torch.zeros((layer_num, self.num_pages, self.bytes_per_page), dtype=torch.uint8, device=device)
        self.HOLD_TOKEN_MEMINDEX = size

    def get_layer_buffer(self, layer_index: int) -> torch.Tensor:
        return self.buffer[layer_index]

    def _loc_offsets(self, loc: torch.Tensor):
        loc = loc.long()
        page = torch.div(loc, self.page_size, rounding_mode="floor")
        token = loc % self.page_size
        page_base = page * self.bytes_per_page
        data_offsets = page_base + token * self.data_bytes_per_token
        scale_offsets = page_base + self.scale_offset_in_page + token * self.scale_bytes_per_token
        return data_offsets, scale_offsets

    def write(self, layer_index: int, loc: torch.Tensor, packed: torch.Tensor) -> None:
        if loc.numel() == 0:
            return
        loc = loc.reshape(-1)
        packed = packed.reshape(-1, self.bytes_per_token)
        flat = self.buffer[layer_index].view(-1)
        data_offsets, scale_offsets = self._loc_offsets(loc)
        data_range = torch.arange(self.data_bytes_per_token, device=loc.device)
        scale_range = torch.arange(self.scale_bytes_per_token, device=loc.device)
        flat[data_offsets.unsqueeze(1) + data_range.unsqueeze(0)] = packed[:, : self.data_bytes_per_token]
        flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)] = packed[:, self.data_bytes_per_token :]
        return

    def read(self, layer_index: int, loc: torch.Tensor) -> torch.Tensor:
        loc = loc.reshape(-1)
        if loc.numel() == 0:
            return torch.empty((0, self.bytes_per_token), dtype=torch.uint8, device=self.buffer.device)
        flat = self.buffer[layer_index].view(-1)
        data_offsets, scale_offsets = self._loc_offsets(loc)
        data_range = torch.arange(self.data_bytes_per_token, device=loc.device)
        scale_range = torch.arange(self.scale_bytes_per_token, device=loc.device)
        data = flat[data_offsets.unsqueeze(1) + data_range.unsqueeze(0)]
        scale = flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)]
        return torch.cat([data, scale], dim=1)


class DeepseekV4MemoryManager(MemoryManager):
    """DeepSeek-V4 KV cache: 窗口 latent(全层) + c4/c128 压缩 latent(压实层) + c4 indexer-K。

    与兄弟 manager 一致的 token-slot 设计；req 索引的表都在 DeepseekV4ReqManager。

    - ``swa_pool``: packed latent for all layers. The request manager owns its
      physical pages and request-private page table. Prefill retains the entire
      current chunk, without a fixed-size ring that could overwrite unread KV.
    - ``c4_pool``/``c128_pool``: 压缩 latent，按 qwen3next 的层号压实手法只为压缩层建层；
      c4 另带 packed indexer-K 池。统一 256-token 页拥有三个 packed 历史页；
      压缩槽位直接由组末 full slot 除以压缩率得到，随 full 页分配和释放。
    - 写入走模型专用的 fused norm/RoPE packed writer，显式接收本轮 SWA 槽；
      torch codecs 保留为 ABI 的可执行规格(单测 oracle)。
    """

    operator_class = DeepseekV4MemOperator

    mla_nope_dim = DSV4_MLA_NOPE_DIM  # 448
    mla_rope_dim = DSV4_MLA_ROPE_DIM  # 64
    mla_head_dim = DSV4_MLA_HEAD_DIM  # 512
    mla_quant_group_size = DSV4_MLA_QUANT_GROUP_SIZE  # 64
    mla_scale_bytes = DSV4_MLA_SCALE_BYTES  # 8
    mla_bytes_per_token = DSV4_MLA_BYTES_PER_TOKEN  # 584
    indexer_head_dim_default = DSV4_INDEXER_HEAD_DIM  # 128
    indexer_bytes_per_token = DSV4_INDEXER_BYTES_PER_TOKEN  # 132

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        compress_rates: List[int],
        max_request_num: int,
        mtp_step: int,
        indexer_head_dim: int = 128,
        cpu_cache_token_page_size: int = DSV4_CPU_CACHE_TOKEN_PAGE_SIZE,
        swa_full_tokens_ratio: float = DSV4_SWA_FULL_TOKENS_RATIO,
        always_copy=False,
        mem_fraction=0.9,
        memory_reservations=None,
    ):
        if get_env_start_args().page_size != DSV4_PROMPT_CACHE_PAGE_SIZE:
            raise ValueError("DeepSeek-V4 requires --page_size 256")
        assert head_num == 1, "DeepSeek-V4 是 MLA(MQA)，dense latent 的 head_num 必须为 1"
        assert head_dim == self.mla_head_dim, f"DeepSeek-V4 packed KV 期望 head_dim={self.mla_head_dim}"
        assert (
            indexer_head_dim == self.indexer_head_dim_default
        ), f"DeepSeek-V4 packed indexer-K 期望 indexer_head_dim={self.indexer_head_dim_default}"
        assert len(compress_rates) == layer_num, f"compress_rates 长度 {len(compress_rates)} 必须等于 layer_num {layer_num}"
        assert all(r in (0, 4, 128) for r in compress_rates), "compress_rates 取值只能是 0/4/128"
        assert max_request_num > 0, "max_request_num 必须为正数"
        assert 0 <= mtp_step < DSV4_C128_STATE_RING, "mtp_step 必须位于 [0, 128)"

        self.compress_rates = list(compress_rates)
        self.n_c4 = sum(1 for r in self.compress_rates if r == 4)
        self.n_c128 = sum(1 for r in self.compress_rates if r == 128)
        self.indexer_head_dim = indexer_head_dim
        self.max_request_num = max_request_num
        self.c4_state_ring = DSV4_C4_STATE_RING + mtp_step
        self.c128_state_ring = _ceil_div(DSV4_C128_STATE_RING + mtp_step, 4) * 4
        self.swa_full_tokens_ratio = float(swa_full_tokens_ratio)
        self.cpu_cache_layout = DeepseekV4CpuCacheLayout.from_compress_rates(
            self.compress_rates,
            token_page_size=cpu_cache_token_page_size,
            head_dim=head_dim,
            indexer_head_dim=indexer_head_dim,
        )

        # 全局层号 -> 各压缩池内的压实层号(同 qwen3next 的层号压实手法)
        self.layer_to_c4_idx = {}
        self.layer_to_c128_idx = {}
        c4 = c128 = 0
        for lid, r in enumerate(self.compress_rates):
            if r == 4:
                self.layer_to_c4_idx[lid] = c4
                c4 += 1
            elif r == 128:
                self.layer_to_c128_idx[lid] = c128
                c128 += 1

        super().__init__(
            size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            always_copy,
            mem_fraction,
            memory_reservations=memory_reservations,
        )

    # ------------------------------------------------------------------ sizing
    def _planned_swa_size(self, full_size: int) -> int:
        return _ceil_div(int(full_size * self.swa_full_tokens_ratio), DSV4_SWA_PAGE_SIZE) * DSV4_SWA_PAGE_SIZE

    @staticmethod
    def _paged_state_rows(num_swa_pages: int, ring: int, ratio: int) -> int:
        rows = num_swa_pages * ring + ring + 1
        return _ceil_div(rows, ratio) * ratio

    @staticmethod
    def _init_state_sentinel(buffer: torch.Tensor) -> None:
        half = buffer.shape[-1] // 2
        buffer[:, -1, :half].zero_()
        buffer[:, -1, half:].fill_(float("-inf"))
        return

    def get_cell_size(self):
        kv_bytes = self.mla_bytes_per_token
        indexer_bytes = self.indexer_bytes_per_token
        state_dtype_bytes = torch._utils._element_size(torch.float32)
        c4_state_width = 4 * self.head_dim + 4 * self.indexer_head_dim
        c4_state_bytes = self.c4_state_ring / DSV4_SWA_PAGE_SIZE * c4_state_width * state_dtype_bytes * self.n_c4
        swa_slot = kv_bytes * self.layer_num + c4_state_bytes
        compressed = (kv_bytes + indexer_bytes) * self.n_c4 / 4 + kv_bytes * self.n_c128 / 128

        return swa_slot * self.swa_full_tokens_ratio + compressed

    def get_fixed_memory_size(self):
        state_rows = (self.max_request_num + 1) * self.c128_state_ring + 1
        return self.n_c128 * state_rows * (2 * self.head_dim) * torch._utils._element_size(torch.float32)

    def get_pd_kv_move_buffer_size(self):
        args = get_env_start_args()
        if args.run_mode not in ["prefill", "decode"]:
            return 0
        layout = DeepseekV4PDCacheLayout.from_compress_rates(
            self.compress_rates,
            token_page_size=args.pd_kv_page_size,
            head_dim=self.head_dim,
            indexer_head_dim=self.indexer_head_dim,
        )
        return args.pd_kv_page_num * layout.page_nbytes

    # ------------------------------------------------------------------ buffers
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        rank_in_node = get_current_rank_in_node()
        server = get_unique_server_name()

        self.swa_size = self._planned_swa_size(size)
        self.swa_pool = PackedPagePool(
            size=self.swa_size,
            page_size=DSV4_SWA_PAGE_SIZE,
            layer_num=layer_num,
            data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
            scale_bytes=self.mla_scale_bytes,
            align_bytes=DSV4_MLA_PAGE_ALIGN_BYTES,
        )
        # 注意: 该别名是 page 索引([layer, num_pages, bytes_per_page])而非 token 索引，
        # 只允许 get_att_input_params 的消费者使用；token 索引语义的继承接口已显式 fence。
        self.kv_buffer = self.swa_pool.buffer
        # 页粒度分配(页 = 128 槽,位置对齐): 槽位不变式 slot(p) = page_base + p%128。
        # swa_size 整页对齐 ⇒ HOLD 槽(swa_size)独占池子最后一个物理页,永不参与分配。
        self.swa_num_pages = self.swa_size // DSV4_SWA_PAGE_SIZE
        self.swa_page_allocator = KvCacheAllocator(
            self.swa_num_pages, shared_name=f"{server}_dsv4_swa_can_use_page_num_{rank_in_node}"
        )
        self.HOLD_TOKEN_MEMINDEX = size

        self.c4_size = _ceil_div(size, 4)
        self.c128_size = _ceil_div(size, 128)
        self.c4_pool: Optional[PackedPagePool] = None
        self.c4_indexer_pool: Optional[PackedPagePool] = None
        self.c128_pool: Optional[PackedPagePool] = None
        self.c4_state_buffer: Optional[torch.Tensor] = None
        self.c4_indexer_state_buffer: Optional[torch.Tensor] = None
        self.c128_state_buffer: Optional[torch.Tensor] = None
        if self.n_c4 > 0:
            self.c4_pool = PackedPagePool(
                size=self.c4_size,
                page_size=DSV4_C4_PAGE_SIZE,
                layer_num=self.n_c4,
                data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                scale_bytes=self.mla_scale_bytes,
                align_bytes=DSV4_MLA_PAGE_ALIGN_BYTES,
            )
            self.c4_indexer_pool = PackedPagePool(
                size=self.c4_size,
                page_size=DSV4_C4_PAGE_SIZE,
                layer_num=self.n_c4,
                data_bytes=self.indexer_head_dim,
                scale_bytes=DSV4_INDEXER_SCALE_BYTES,
            )
            # c4 compressor 在途状态(attention + indexer): swa 页派生寻址(翻译③),随 swa 页
            # 生灭；radix 命中从 CPU checkpoint 恢复。行数 = 页数*ring + ring(HOLD 页) + 1(哨兵),
            # 取整到 ratio;末行哨兵 kv=0/score=-inf(KVAndScore.clear 语义),其余行由内核在
            # 组起点覆写,无需按页清零。last_dim = 2*coff*head_dim(overlap coff=2)。
            state_rows = self._paged_state_rows(self.swa_num_pages, self.c4_state_ring, 4)
            self.c4_state_buffer = torch.zeros(
                (self.n_c4, state_rows, 4 * self.head_dim), dtype=torch.float32, device="cuda"
            )
            self.c4_indexer_state_buffer = torch.zeros(
                (self.n_c4, state_rows, 4 * self.indexer_head_dim), dtype=torch.float32, device="cuda"
            )
            for buf in (self.c4_state_buffer, self.c4_indexer_state_buffer):
                self._init_state_sentinel(buf)
        if self.n_c128 > 0:
            self.c128_pool = PackedPagePool(
                size=self.c128_size,
                page_size=DSV4_C128_PAGE_SIZE,
                layer_num=self.n_c128,
                data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                scale_bytes=self.mla_scale_bytes,
                align_bytes=DSV4_MLA_PAGE_ALIGN_BYTES,
            )
            # c128 compressor 在途状态按 request 寻址。每个 request 保留完整的 128-token
            # 聚合窗口以及 MTP 候选余量；最后一行供无效请求/位置读取哨兵。
            state_rows = (self.max_request_num + 1) * self.c128_state_ring + 1
            self.c128_state_buffer = torch.zeros(
                (self.n_c128, state_rows, 2 * self.head_dim), dtype=torch.float32, device="cuda"
            )
            self._init_state_sentinel(self.c128_state_buffer)

        layout = self.cpu_cache_layout
        assert self.swa_pool.bytes_per_page == layout.swa_gpu_page_nbytes
        if self.n_c4:
            assert self.c4_pool.bytes_per_page == layout.c4_gpu_page_nbytes
            assert self.c4_indexer_pool.bytes_per_page == layout.c4_indexer_gpu_page_nbytes
        assert layout.c128_row_nbytes == self.mla_bytes_per_token

        from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager

        args = get_env_start_args()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        self.big_page_buffers = DeepseekV4StateCacheManager(
            size=_ceil_div(size, big_page_tokens),
            layout=layout,
        )

        logger.info(
            f"DeepseekV4MemoryManager pools: full_tokens={size} swa={self.swa_size}({self.swa_num_pages}p) "
            f"c4={self.c4_size}(L={self.n_c4}) c128={self.c128_size}(L={self.n_c128}) "
            f"c4_state_ring={self.c4_state_ring} c128_state_ring={self.c128_state_ring} "
            f"max_requests={self.max_request_num} "
            f"packed_kv_bytes={self.mla_bytes_per_token} indexer_bytes={self.indexer_bytes_per_token}"
        )

    # ------------------------------------------------------------------ buffer accessors
    def get_att_input_params(self, layer_index: int):
        return self.swa_pool.get_layer_buffer(layer_index)

    def _pool_and_local_layer(self, layer_index: int):
        r = self.compress_rates[layer_index]
        if r == 4:
            return self.c4_pool, self.layer_to_c4_idx[layer_index]
        if r == 128:
            return self.c128_pool, self.layer_to_c128_idx[layer_index]
        raise AssertionError(f"layer {layer_index} (rate {r}) 不是压缩层，没有压缩池")

    def get_compressed_kv_buffer(self, layer_index: int) -> torch.Tensor:
        pool, local_layer = self._pool_and_local_layer(layer_index)
        return pool.get_layer_buffer(local_layer)

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 indexer-K"
        return self.c4_indexer_pool.get_layer_buffer(self.layer_to_c4_idx[layer_index])

    def get_c4_state_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 paged compressor state"
        return self.c4_state_buffer[self.layer_to_c4_idx[layer_index]]

    def get_c4_indexer_state_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 paged indexer state"
        return self.c4_indexer_state_buffer[self.layer_to_c4_idx[layer_index]]

    def get_c128_state_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.compress_rates[layer_index] == 128, "只有 c128(HCA) 层有 request-scoped compressor state"
        return self.c128_state_buffer[self.layer_to_c128_idx[layer_index]]

    # ------------------------------------------------------------------ CPU cache load resources
    def get_loadable_cpu_cache_end(
        self,
        loaded_start: int,
        requested_end: int,
        full_token_capacity: int,
        swa_page_capacity: int,
    ) -> int:
        """Return the farthest loadable checkpoint boundary, or zero.

        History is independently addressable in 256-token blocks, but resume
        SWA/state exists only at the end of each CPU checkpoint page.  Therefore
        capacity cropping must never return an intermediate 256-token boundary.
        """
        loaded_start = int(loaded_start)
        requested_end = int(requested_end)
        page = self.cpu_cache_layout.token_page_size
        if loaded_start < 0 or loaded_start % DSV4_PROMPT_CACHE_PAGE_SIZE != 0:
            raise ValueError(f"DeepSeek-V4 CPU cache loaded_start must be 256-token aligned, got {loaded_start}")
        if requested_end <= loaded_start or requested_end % page != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache requested_end must be a checkpoint boundary after loaded_start, "
                f"got start={loaded_start}, end={requested_end}, page={page}"
            )
        if int(swa_page_capacity) < 2:
            return 0

        token_capacity = int(full_token_capacity)
        token_capacity = token_capacity // DSV4_PROMPT_CACHE_PAGE_SIZE * DSV4_PROMPT_CACHE_PAGE_SIZE
        loadable_end = min(requested_end, (loaded_start + token_capacity) // page * page)
        return loadable_end if loadable_end > loaded_start else 0

    def prepare_cpu_cache_load(
        self, token_num: int, loaded_end: int, resume_swa_slots: torch.Tensor
    ) -> DeepseekV4CpuCacheLoadPlan:
        """Allocate a missing history suffix using the request's continuation slots.

        ``loaded_end`` is the CPU checkpoint boundary.  ``token_num`` may be
        smaller than the checkpoint page when a GPU radix prefix overlaps its
        beginning, but both endpoints remain 256-token aligned.
        """
        token_num = int(token_num)
        loaded_end = int(loaded_end)
        layout = self.cpu_cache_layout
        if token_num <= 0 or token_num % DSV4_PROMPT_CACHE_PAGE_SIZE != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache load size must be a positive multiple of "
                f"{DSV4_PROMPT_CACHE_PAGE_SIZE}, got {token_num}"
            )
        if loaded_end < token_num or loaded_end % layout.token_page_size != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache loaded_end must be a checkpoint boundary >= token_num, "
                f"got loaded_end={loaded_end}, token_num={token_num}, page={layout.token_page_size}"
            )

        block_num = token_num // DSV4_PROMPT_CACHE_PAGE_SIZE
        device = self.swa_pool.buffer.device
        full_indexes_cpu = self.alloc(token_num)

        mem_indexes = full_indexes_cpu.to(device, non_blocking=True)
        history_full_slots = mem_indexes.view(block_num, DSV4_PROMPT_CACHE_PAGE_SIZE)
        history_c4_slots = history_full_slots[:, 3::4] // 4 if self.n_c4 else None
        history_c128_slots = history_full_slots[:, 127::128] // 128 if self.n_c128 else None

        return DeepseekV4CpuCacheLoadPlan(
            loaded_start=loaded_end - token_num,
            loaded_end=loaded_end,
            mem_indexes=mem_indexes,
            history_full_slots=history_full_slots,
            history_c4_slots=history_c4_slots,
            history_c128_slots=history_c128_slots,
            resume_swa_slots=resume_swa_slots,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Pinned CPU checkpoints are process-local; IPC readers need only GPU storage.
        state["big_page_buffers"] = None
        return state

    def alloc_dspark_swa_block(self, token_num: int, block_size: int):
        """Assign one temporary SWA scratch page to each DSpark proposal block.

        Proposal KV is consumed within the draft forward, then its scratch
        pages are released. Request token pages stay reserved. Keeping the whole
        block in a private page avoids the host-side sequence-length decision
        required by the position-aligned target cache.  Attention still uses
        the absolute positions stored in the request table; physical SWA slots
        only identify the packed KV rows.
        """
        block_size = int(block_size)
        assert block_size > 0
        assert block_size <= DSV4_SWA_PAGE_SIZE
        assert token_num % block_size == 0

        req_num = token_num // block_size
        if req_num == 0:
            return (
                torch.empty((0,), dtype=torch.int32, device="cpu"),
                torch.empty((0,), dtype=torch.int32, device=self.swa_pool.buffer.device),
            )

        device = self.swa_pool.buffer.device
        pages_cpu = self.swa_page_allocator.alloc(req_num)
        pages = pages_cpu.to(device, non_blocking=True)
        # The proposal block addresses its own page directly, outside the request page table.
        return pages_cpu, pages

    def free_dspark_swa_block(self, pages_cpu: torch.Tensor) -> None:
        """Release only proposal scratch; token pages remain owned by the request."""
        self.swa_page_allocator.free(pages_cpu)

    def free_all(self):
        super().free_all()
        self.swa_page_allocator.free_all()
        self.big_page_buffers.clear_to_init_state()

    # ------------------------------------------------------------------ packed codecs (torch reference)
    # 与 sglang/vllm 的 fp8_ds_mla 字节布局逐位对齐(ue8m0 幂次 scale)。这些 torch 实现是该 ABI 的
    # 可执行规格(单测 oracle，triton writer 与其逐字节对拍)，不可删除。
    # torch参考，未使用
    def _pack_mla_kv(self, kv: torch.Tensor) -> torch.Tensor:
        kv = kv.reshape(-1, self.mla_head_dim)
        out = torch.empty((kv.shape[0], self.mla_bytes_per_token), dtype=torch.uint8, device=kv.device)
        nope = kv[:, : self.mla_nope_dim].float().reshape(-1, self.mla_scale_bytes - 1, self.mla_quant_group_size)
        amax = torch.clamp(nope.abs().amax(dim=-1), min=DSV4_FP8_AMAX_MIN)
        scale = amax / DSV4_FP8_E4M3_MAX
        scale_exp = torch.ceil(torch.log2(scale)).to(torch.int32)
        scale = torch.exp2(scale_exp.float())
        nope_fp8 = torch.clamp(nope / scale.unsqueeze(-1), -DSV4_FP8_E4M3_MAX, DSV4_FP8_E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        out[:, : self.mla_nope_dim].copy_(nope_fp8.reshape(-1, self.mla_nope_dim).view(dtype=torch.uint8))
        rope_start = self.mla_nope_dim
        rope_end = rope_start + self.mla_rope_dim * 2
        rope = kv[:, self.mla_nope_dim : self.mla_head_dim].to(torch.bfloat16)
        out[:, rope_start:rope_end].copy_(rope.view(dtype=torch.uint8).reshape(-1, self.mla_rope_dim * 2))
        scale_start = rope_end
        scale_end = scale_start + self.mla_scale_bytes - 1
        out[:, scale_start:scale_end].copy_((scale_exp + 127).to(torch.uint8))
        out[:, scale_end].zero_()
        return out

    def _unpack_mla_kv(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(-1, self.mla_bytes_per_token)
        if packed.shape[0] == 0:
            return torch.empty((0, self.mla_head_dim), dtype=self.dtype, device=packed.device)
        nope_fp8 = packed[:, : self.mla_nope_dim].view(dtype=torch.float8_e4m3fn).float()
        nope_fp8 = nope_fp8.reshape(-1, self.mla_scale_bytes - 1, self.mla_quant_group_size)
        rope_start = self.mla_nope_dim
        rope_end = rope_start + self.mla_rope_dim * 2
        scale_start = rope_end
        scale_end = scale_start + self.mla_scale_bytes - 1
        scale_exp = packed[:, scale_start:scale_end].to(torch.int32) - 127
        scale = torch.exp2(scale_exp.float())
        nope = (nope_fp8 * scale.reshape(-1, self.mla_scale_bytes - 1, 1)).reshape(-1, self.mla_nope_dim)
        rope = packed[:, rope_start:rope_end].view(dtype=torch.bfloat16)
        return torch.cat([nope.to(self.dtype), rope.to(self.dtype)], dim=-1)

    def _pack_indexer_k(self, indexer_k: torch.Tensor) -> torch.Tensor:
        indexer_k = indexer_k.reshape(-1, self.indexer_head_dim)
        out = torch.empty(
            (indexer_k.shape[0], self.indexer_bytes_per_token),
            dtype=torch.uint8,
            device=indexer_k.device,
        )
        k_float = indexer_k.float()
        amax = torch.clamp(k_float.abs().amax(dim=-1, keepdim=True), min=DSV4_FP8_AMAX_MIN)
        scale = amax / DSV4_FP8_E4M3_MAX
        k_fp8 = torch.clamp(k_float / scale, -DSV4_FP8_E4M3_MAX, DSV4_FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        out[:, : self.indexer_head_dim].copy_(k_fp8.view(dtype=torch.uint8))
        out[:, self.indexer_head_dim :].copy_(scale.view(dtype=torch.uint8).reshape(-1, DSV4_INDEXER_SCALE_BYTES))
        return out

    def _unpack_indexer_k(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(-1, self.indexer_bytes_per_token)
        if packed.shape[0] == 0:
            return torch.empty((0, self.indexer_head_dim), dtype=self.dtype, device=packed.device)
        k_fp8 = packed[:, : self.indexer_head_dim].view(dtype=torch.float8_e4m3fn).float()
        scale = packed[:, self.indexer_head_dim :].view(dtype=torch.float32)
        return (k_fp8 * scale).to(self.dtype)

    # ------------------------------------------------------------------ cache write paths
    def pack_mla_kv_to_cache_fused_norm_rope(
        self,
        layer_index: int,
        swa_slots: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Fuse RMSNorm and RoPE into the packed writer at request-owned SWA slots."""
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope_cuda import (
            fused_k_norm_rope_flashmla,
        )

        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_slots,
            kvcache=self.swa_pool.get_layer_buffer(layer_index),
            page_size=self.swa_pool.page_size,
        )
        return

    def pack_compressed_kv_to_cache(self, layer_index: int, slots: torch.Tensor, comp: torch.Tensor):
        if comp.shape[0] == 0:
            return
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_kv_flashmla_dsv4 import (
            destindex_copy_kv_flashmla_dsv4,
        )

        pool, local_layer = self._pool_and_local_layer(layer_index)
        destindex_copy_kv_flashmla_dsv4(
            comp.reshape(-1, self.mla_head_dim),
            slots.to(comp.device),
            pool.get_layer_buffer(local_layer),
            pool.page_size,
        )

    def pack_indexer_k_to_cache(
        self,
        layer_index: int,
        mem_index: torch.Tensor,
        positions: torch.Tensor,
        indexer_k: torch.Tensor,
    ):
        if indexer_k.shape[0] == 0:
            return
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 indexer-K"
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import (
            destindex_copy_indexer_k_dsv4,
        )

        destindex_copy_indexer_k_dsv4(
            indexer_k.reshape(-1, self.indexer_head_dim),
            mem_index.reshape(-1),
            positions.reshape(-1),
            self.c4_indexer_pool.get_layer_buffer(self.layer_to_c4_idx[layer_index]),
            self.c4_indexer_pool.page_size,
        )

    # ------------------------------------------------------------------ fenced inherited APIs
    # kv_buffer 是 page 索引的 uint8 buffer，基类按 token 索引读写的接口会静默写坏数据，显式拦截。
    def get_index_kv_buffer(self, index):
        raise NotImplementedError("DeepSeek-V4 packed page cache does not support token-indexed kv_buffer io")

    def load_index_kv_buffer(self, index, load_tensor_dict):
        raise NotImplementedError("DeepSeek-V4 packed page cache does not support token-indexed kv_buffer io")

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        self.pd_cache_layout = DeepseekV4PDCacheLayout.from_compress_rates(
            self.compress_rates,
            token_page_size=page_size,
            head_dim=self.head_dim,
            indexer_head_dim=self.indexer_head_dim,
        )
        self.kv_move_buffer = torch.empty(
            (page_num, 1, 1, 1, self.pd_cache_layout.page_nbytes),
            dtype=torch.uint8,
            device="cuda",
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty((page_size,), dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        start_kv_index: int,
        request_kv_len: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        assert page_kind == "kv"
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        from lightllm.models.deepseek_v4.triton_kernel.pd_cache_io import pack_pd_cache_page

        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        assert len(dp_mems) == dp_world_size
        return pack_pd_cache_page(
            dp_mems[0],
            self.pd_cache_layout,
            mem_indexes_gpu,
            self.kv_move_buffer[page_index],
            start_kv_index,
            request_kv_len,
            req_idx,
        )

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        start_kv_index: int,
        request_kv_len: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        assert page_kind == "kv"
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][: len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        from lightllm.models.deepseek_v4.triton_kernel.pd_cache_io import unpack_pd_cache_page

        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        assert len(dp_mems) == dp_world_size
        for mem_manager in dp_mems:
            unpack_pd_cache_page(
                mem_manager,
                self.pd_cache_layout,
                mem_indexes_gpu,
                self.kv_move_buffer[page_index],
                start_kv_index,
                request_kv_len,
                req_idx,
            )
        return
