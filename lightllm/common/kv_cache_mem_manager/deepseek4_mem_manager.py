import torch
from typing import List, Optional, Union
from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from .allocator import KvCacheAllocator
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


# fp8_ds_mla packed-latent byte layout (ABI shared with the flash_mla extra-cache fork and
# sglang/vllm): 448B NoPE fp8 + 64*2B RoPE bf16 + 7B ue8m0 scale + 1B pad = 584B per token,
# stored in page slabs whose tail carries the per-token scale bytes.
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
DSV4_FP8_SCALE_MIN = 1e-4  # 1e-4
DSV4_SWA_PAGE_SIZE = 128  # 128 slots/page
DSV4_C4_PAGE_SIZE = 64  # 64 slots/page
DSV4_C128_PAGE_SIZE = 2  # 2 slots/page
DSV4_PROMPT_CACHE_PAGE_SIZE = DSV4_C4_PAGE_SIZE * 4  # 256 (= c4 ratio)
# compressor state ring: c4 overlap 对为每页 2 个分组槽 × ratio 4 行;c128 离线聚合为每页 1 组。
DSV4_C4_STATE_RING = 8  # 8 rows/page
DSV4_C128_STATE_RING = 128  # 128 rows/page
# swa 池占 full token 空间的比例(sglang DSV4 默认 swa_full_tokens_ratio=0.1 同值)。
# 瞬时借页/驱逐走 swa 压力阀;池子大小仅按 ratio 切分,不再叠加结构性余量。
DSV4_SWA_FULL_TOKENS_RATIO = 0.1  # 0.1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class PackedPagePool:
    """fp8_ds_mla 风格的 page-slab 存储: 每页前段连续放 token 的 data 字节，页尾放 per-token scale 字节。

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
        packed = packed.reshape(-1, self.bytes_per_token).contiguous()
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
        return torch.cat([data, scale], dim=1).contiguous()


class DeepseekV4MemoryManager(MemoryManager):
    """DeepSeek-V4 KV cache: 窗口 latent(全层) + c4/c128 压缩 latent(压实层) + c4 indexer-K。

    与兄弟 manager 一致的 token-slot 设计；req 索引的表都在 DeepseekV4ReqManager。

    - ``swa_pool``: 584B packed latent，所有层。池子小于 full token 空间；prep 阶段
      ``alloc_swa_prefill/decode`` 按**页**(128 槽,位置对齐: slot(p)=page_base+p%128)分配，
      映射记录到 ``full_to_swa_indexs``(以 full token 槽位为键)。出窗槽位由 DeepseekV4ReqManager
      在 prep 阶段批量惰性回收(``evict_swa``,页存活计数减到 0 才整页归还)；full 槽位释放时
      ``free`` 级联回收对应 swa 槽，所以 radix 驱逐/请求释放/暂停无需任何额外协议。
      页 allocator 触底时先走 swa free hook(radix 对 ref==0 节点 free)再 assert。
      没有 ring buffer，prefill chunk 大小不受 sliding_window 限制。
    - ``c4_pool``/``c128_pool``: 压缩 latent，按 qwen3next 的层号压实手法只为压缩层建层；
      c4 另带 packed indexer-K 池。槽位映射(``full_to_c4/c128_indexs``)以组末 token 的 full
      槽位为键(prep 阶段分配/scatter)，``free`` 级联回收，与 swa 完全同构。
    - 写入走标准 operator 路径(``pack_mla_kv_to_cache``)，内部为 triton packed writer；
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
        indexer_head_dim: int = 128,
        max_request_num: Optional[int] = None,
        sliding_window: Optional[int] = None,
        swa_full_tokens_ratio: float = DSV4_SWA_FULL_TOKENS_RATIO,
        always_copy=False,
        mem_fraction=0.9,
    ):
        assert head_num == 1, "DeepSeek-V4 是 MLA(MQA)，dense latent 的 head_num 必须为 1"
        assert head_dim == self.mla_head_dim, f"DeepSeek-V4 packed KV 期望 head_dim={self.mla_head_dim}"
        assert (
            indexer_head_dim == self.indexer_head_dim_default
        ), f"DeepSeek-V4 packed indexer-K 期望 indexer_head_dim={self.indexer_head_dim_default}"
        assert len(compress_rates) == layer_num, f"compress_rates 长度 {len(compress_rates)} 必须等于 layer_num {layer_num}"
        assert all(r in (0, 4, 128) for r in compress_rates), "compress_rates 取值只能是 0/4/128"

        self.compress_rates = list(compress_rates)
        self.n_c4 = sum(1 for r in self.compress_rates if r == 4)
        self.n_c128 = sum(1 for r in self.compress_rates if r == 128)
        self.indexer_head_dim = indexer_head_dim
        self.max_request_num = max_request_num
        self.sliding_window = sliding_window
        self.swa_full_tokens_ratio = float(swa_full_tokens_ratio)

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

        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

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
        c128_state_width = 2 * self.head_dim
        c4_state_bytes = DSV4_C4_STATE_RING / DSV4_SWA_PAGE_SIZE * c4_state_width * state_dtype_bytes * self.n_c4
        c128_state_bytes = (
            DSV4_C128_STATE_RING / DSV4_SWA_PAGE_SIZE * c128_state_width * state_dtype_bytes * self.n_c128
        )
        swa_slot = kv_bytes * self.layer_num + c4_state_bytes + c128_state_bytes
        compressed = (kv_bytes + indexer_bytes) * self.n_c4 / 4 + kv_bytes * self.n_c128 / 128

        return swa_slot * self.swa_full_tokens_ratio + compressed

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
        # 页存活计数 = 指向该页的有效 full_to_swa 行数;减到 0 归还 allocator(出窗逐 token
        # 回收下,「部分出窗页」计数 > 0 自然受保护)。下标含 HOLD 页(只读不增减)。
        self.swa_page_live_count = torch.zeros((self.swa_pool.num_pages,), dtype=torch.int32, device="cuda")
        # swa free hook(可选): 页 allocator 触底时回调(radix 对 ref==0 节点 free swa 页),
        # 由 backend 在 radix cache 创建后 register;assert 仍是最后防线。
        self._free_radix_unreferenced_swa_fn = None
        self.full_to_swa_indexs = torch.full((size + 1,), -1, dtype=torch.int32, device="cuda")
        self.full_to_swa_indexs[size] = self.swa_pool.HOLD_TOKEN_MEMINDEX

        self.c4_size = _ceil_div(size, 4)
        self.c128_size = _ceil_div(size, 128)
        self.c4_pool: Optional[PackedPagePool] = None
        self.c4_indexer_pool: Optional[PackedPagePool] = None
        self.c4_allocator: Optional[KvCacheAllocator] = None
        self.c4_page_allocator: Optional[KvCacheAllocator] = None
        self.c4_page_live_count: Optional[torch.Tensor] = None
        self.c128_pool: Optional[PackedPagePool] = None
        self.c128_allocator: Optional[KvCacheAllocator] = None
        self.c4_state_buffer: Optional[torch.Tensor] = None
        self.c4_indexer_state_buffer: Optional[torch.Tensor] = None
        self.c128_state_buffer: Optional[torch.Tensor] = None
        # 压缩槽映射: 键 = 组末 token(位置 (g+1)%ratio==0)的 full 槽位,值 = 压缩池槽位。
        # 与 full_to_swa_indexs 同构: radix 持有 full 槽 => 映射行存活,free 级联回收。
        self.full_to_c4_indexs: Optional[torch.Tensor] = None
        self.full_to_c128_indexs: Optional[torch.Tensor] = None
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
            self.c4_num_pages = self.c4_size // DSV4_C4_PAGE_SIZE
            assert self.c4_num_pages > 0, "DeepSeek-V4 c4 pool must have at least one usable full page"
            self.c4_page_allocator = KvCacheAllocator(
                self.c4_num_pages, shared_name=f"{server}_dsv4_c4_can_use_page_num_{rank_in_node}"
            )
            self.c4_page_live_count = torch.zeros((self.c4_pool.num_pages,), dtype=torch.int32, device="cuda")
            self.full_to_c4_indexs = torch.full((size + 1,), -1, dtype=torch.int32, device="cuda")
            self.full_to_c4_indexs[size] = self.c4_pool.HOLD_TOKEN_MEMINDEX
            # c4 compressor 在途状态(attention + indexer): swa 页派生寻址(翻译③),随 swa 页
            # 生灭 -> radix 命中零拷贝续算。行数 = 页数*ring + ring(HOLD 页) + 1(哨兵),
            # 取整到 ratio;末行哨兵 kv=0/score=-inf(KVAndScore.clear 语义),其余行由内核在
            # 组起点覆写,无需按页清零。last_dim = 2*coff*head_dim(overlap coff=2)。
            state_rows = self._paged_state_rows(self.swa_num_pages, DSV4_C4_STATE_RING, 4)
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
            self.c128_allocator = KvCacheAllocator(
                self.c128_size, shared_name=f"{server}_dsv4_c128_can_use_token_num_{rank_in_node}"
            )
            self.full_to_c128_indexs = torch.full((size + 1,), -1, dtype=torch.int32, device="cuda")
            self.full_to_c128_indexs[size] = self.c128_pool.HOLD_TOKEN_MEMINDEX
            # c128 compressor 在途状态: 与 c4 同样由 full->swa 推导行号,但 ring=128 且无 overlap。
            # last_dim = 2*head_dim;末行是 swa 缺失/出窗时读取的哨兵。
            state_rows = self._paged_state_rows(self.swa_num_pages, DSV4_C128_STATE_RING, 128)
            self.c128_state_buffer = torch.zeros(
                (self.n_c128, state_rows, 2 * self.head_dim), dtype=torch.float32, device="cuda"
            )
            self._init_state_sentinel(self.c128_state_buffer)

        logger.info(
            f"DeepseekV4MemoryManager pools: full_tokens={size} swa={self.swa_size}({self.swa_num_pages}p) "
            f"c4={self.c4_size}(L={self.n_c4}) c128={self.c128_size}(L={self.n_c128}) "
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
        assert self.compress_rates[layer_index] == 128, "只有 c128(HCA) 层有 paged compressor state"
        return self.c128_state_buffer[self.layer_to_c128_idx[layer_index]]

    # ------------------------------------------------------------------ swa slot lifecycle
    def register_swa_free_hook(self, fn) -> None:
        """fn(need_pages): 在页 allocator 不足时尝试腾页(radix 对 ref==0 节点 free swa)。"""
        self._free_radix_unreferenced_swa_fn = fn
        return

    def _alloc_swa_pages(self, need_pages: int) -> torch.Tensor:
        if need_pages > self.swa_page_allocator.can_use_mem_size and self._free_radix_unreferenced_swa_fn is not None:
            self._free_radix_unreferenced_swa_fn(need_pages - self.swa_page_allocator.can_use_mem_size)
        return self.swa_page_allocator.alloc(need_pages)

    def _update_swa_page_counts(self, swa_slots: torch.Tensor, delta: int) -> torch.Tensor:
        """按 slot 所在页更新存活计数，返回逐 slot 的页号。"""
        pages = torch.div(swa_slots, DSV4_SWA_PAGE_SIZE, rounding_mode="floor")
        ones = torch.full(pages.shape, delta, dtype=torch.int32, device=pages.device)
        self.swa_page_live_count.index_add_(0, pages, ones)
        return pages

    def alloc_swa_prefill(
        self,
        mem_indexes: torch.Tensor,
        req_to_token_indexs: torch.Tensor,
        req_list: List[int],
        ready_list: List[int],
        seq_list: List[int],
    ) -> None:
        """prefill prep: 为各请求位置 [ready, seq) 的新 token 分配位置对齐的 swa 槽。

        槽位不变式: slot(p) = page_base(p 所在页) + p%128,page_base % 128 == 0。
        续页(start 非整页,只可能是首页)的 base 从上一 token 的映射派生
        (full_to_swa[req_to_token[req, start-1]],该 token 必在保留窗内);其余页全新分配。
        radix 命中(ready 必 128 对齐)的借用方从全新页开始,与节点持有页天然不相交。
        当前 chunk 的 full 槽直接来自 generic preprocess 分配的 mem_indexes，因此不依赖
        req_to_token_indexs 已完成当前 chunk 的 scatter；只有续页的上一 token 查询旧 req 行。
        """
        page = DSV4_SWA_PAGE_SIZE
        hold_req_id = self.max_request_num  # padding 行的请求 id(req_manager.HOLD_REQUEST_ID)

        segs = []  # (req_idx, start, end, mem_offset, n_new_pages, has_cont_page)
        total_new_pages = 0
        mem_offset = 0
        for req_idx, start, end in zip(req_list, ready_list, seq_list):
            q_len = end - start
            if req_idx == hold_req_id or end <= start:
                mem_offset += q_len
                continue
            first_new_page = _ceil_div(start, page)
            n_new = max(0, (end - 1) // page - first_new_page + 1)
            segs.append((req_idx, start, end, mem_offset, n_new, start % page != 0))
            total_new_pages += n_new
            mem_offset += q_len
        if not segs:
            return

        device = self.full_to_swa_indexs.device
        mem_indexes = mem_indexes.reshape(-1)
        new_pages = self._alloc_swa_pages(total_new_pages).to(device, non_blocking=True) if total_new_pages else None
        page_cursor = 0
        for req_idx, start, end, mem_start, n_new, has_cont in segs:
            positions = torch.arange(start, end, dtype=torch.int32, device=device)
            page_local = torch.div(positions, page, rounding_mode="floor") - start // page
            bases = torch.empty(((end - 1) // page - start // page + 1,), dtype=torch.int32, device=device)
            if has_cont:
                prev_slot = self.full_to_swa_indexs[req_to_token_indexs[req_idx, start - 1]]
                bases[0] = prev_slot - (start - 1) % page
            if n_new:
                bases[1 if has_cont else 0 :] = new_pages[page_cursor : page_cursor + n_new] * page
                page_cursor += n_new
            slots = bases[page_local] + positions % page
            full_slots = mem_indexes[mem_start : mem_start + end - start]
            self.full_to_swa_indexs[full_slots] = slots
            self._update_swa_page_counts(slots, 1)
        return

    def alloc_swa_decode(
        self,
        req_list: List[int],
        seq_list: List[int],
        mem_indexes: torch.Tensor,
        prev_full_indexes: torch.Tensor,
    ) -> None:
        """decode prep: 本步 token(位置 seq-1)的 swa 槽。整页起点开新页,否则上一 token 槽 +1
        (位置对齐不变式保证同页连续)。scatter 目标用当前步 mem_indexes。

        调用方传入每行前一 token 的 full 槽；MTP step>0 可直接使用同批前一列。"""
        page = DSV4_SWA_PAGE_SIZE
        hold_req_id = self.max_request_num
        cont_rows, new_rows = [], []
        for i, (req_idx, seq_len) in enumerate(zip(req_list, seq_list)):
            if req_idx == hold_req_id or seq_len <= 0:
                continue
            if (seq_len - 1) % page == 0:
                new_rows.append(i)
            else:
                cont_rows.append(i)
        mem_indexes = mem_indexes.reshape(-1)
        if cont_rows:
            prev_full = prev_full_indexes.reshape(-1)[cont_rows]
            prev_slots = self.full_to_swa_indexs[prev_full]
            slots = prev_slots + 1
            self.full_to_swa_indexs[mem_indexes[cont_rows]] = slots
            self._update_swa_page_counts(slots, 1)
        if new_rows:
            pages = self._alloc_swa_pages(len(new_rows)).to(self.full_to_swa_indexs.device, non_blocking=True)
            slots = pages * page
            self.full_to_swa_indexs[mem_indexes[new_rows]] = slots
            self._update_swa_page_counts(slots, 1)
        return

    def evict_swa(self, full_slots: torch.Tensor) -> None:
        """回收 full 槽位对应的 swa 槽(出窗惰性回收 / free 级联 / 压力阀共用)。
        未映射(-1)的槽位跳过;页计数减到 0 时整页归还 allocator。"""
        if full_slots.numel() == 0:
            return
        full_slots = full_slots.to(self.full_to_swa_indexs.device, non_blocking=True).reshape(-1)
        full_slots = torch.unique(full_slots[full_slots != self.HOLD_TOKEN_MEMINDEX])
        if full_slots.numel() == 0:
            return
        swa_slots = self.full_to_swa_indexs[full_slots]
        valid = swa_slots >= 0
        valid_slots = swa_slots[valid]
        if valid_slots.numel() == 0:
            return
        self.full_to_swa_indexs[full_slots[valid]] = -1
        touched = torch.unique(self._update_swa_page_counts(valid_slots, -1))
        empty = touched[self.swa_page_live_count[touched] == 0]
        if empty.numel() > 0:
            self.swa_page_allocator.free(empty.to(torch.int32))
        return

    def _evict_compress(self, full_slots: torch.Tensor, mapping: torch.Tensor, allocator: KvCacheAllocator) -> None:
        full_slots = full_slots.to(mapping.device, non_blocking=True).reshape(-1)
        # 去重: 同批重复槽会 gather 出重复的压缩槽 -> allocator 双重释放(free 已去重,直呼叫方防御)。
        full_slots = torch.unique(full_slots[full_slots != self.HOLD_TOKEN_MEMINDEX])
        if full_slots.numel() == 0:
            return
        slots = mapping[full_slots]
        valid = slots >= 0
        valid_slots = slots[valid]
        if valid_slots.numel() == 0:
            return
        allocator.free(valid_slots)
        mapping[full_slots[valid]] = -1
        return

    def alloc_c4_pages(self, need_pages: int) -> torch.Tensor:
        assert self.c4_page_allocator is not None, "DeepSeek-V4 c4 page allocator is not initialized"
        return self.c4_page_allocator.alloc(need_pages)

    def count_c4_slots(self, c4_slots: torch.Tensor, delta: int) -> torch.Tensor:
        """按 c4 slot 所在页更新存活计数，返回逐 slot 的页号。"""
        assert self.c4_page_live_count is not None, "DeepSeek-V4 c4 page live count is not initialized"
        pages = torch.div(c4_slots, DSV4_C4_PAGE_SIZE, rounding_mode="floor")
        ones = torch.full(pages.shape, delta, dtype=torch.int32, device=pages.device)
        self.c4_page_live_count.index_add_(0, pages, ones)
        return pages

    def evict_c4(self, full_slots: torch.Tensor) -> None:
        """回收 full 槽位(组末 token)映射的 c4 槽。非组末/未映射(-1)的槽位跳过。"""
        if self.c4_page_allocator is None or full_slots.numel() == 0:
            return
        full_slots = full_slots.to(self.full_to_c4_indexs.device, non_blocking=True).reshape(-1)
        full_slots = torch.unique(full_slots[full_slots != self.HOLD_TOKEN_MEMINDEX])
        if full_slots.numel() == 0:
            return
        slots = self.full_to_c4_indexs[full_slots]
        valid = slots >= 0
        valid_slots = slots[valid]
        if valid_slots.numel() == 0:
            return
        self.full_to_c4_indexs[full_slots[valid]] = -1
        touched = torch.unique(self.count_c4_slots(valid_slots, -1))
        empty = touched[self.c4_page_live_count[touched] == 0]
        if empty.numel() > 0:
            self.c4_page_allocator.free(empty.to(torch.int32))
        return

    def evict_c128(self, full_slots: torch.Tensor) -> None:
        """回收 full 槽位(组末 token)映射的 c128 槽。非组末/未映射(-1)的槽位跳过。"""
        if self.c128_allocator is None or full_slots.numel() == 0:
            return
        self._evict_compress(full_slots, self.full_to_c128_indexs, self.c128_allocator)
        return

    # ------------------------------------------------------------------ alloc/free (cascade)
    def free(self, free_index: Union[torch.Tensor, List[int]]) -> None:
        """释放 full token 槽位，级联回收其 swa 槽与 c4/c128 压缩槽。radix 驱逐、请求释放/暂停都走这里。

        先对 full 槽去重: 同批重复槽位会让映射 gather 出重复的压缩/swa 槽，导致 allocator 双重释放。"""
        if isinstance(free_index, list):
            free_index = torch.tensor(free_index, dtype=torch.int64)
        if free_index.numel() > 0:
            free_index = torch.unique(free_index)
            self.evict_swa(free_index)
            self.evict_c4(free_index)
            self.evict_c128(free_index)
        super().free(free_index)
        return

    def free_all(self):
        super().free_all()
        self.swa_page_allocator.free_all()
        self.swa_page_live_count.zero_()
        self.full_to_swa_indexs.fill_(-1)
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        if self.c4_page_allocator is not None:
            self.c4_page_allocator.free_all()
            self.c4_page_live_count.zero_()
            self.full_to_c4_indexs.fill_(-1)
            self.full_to_c4_indexs[self.HOLD_TOKEN_MEMINDEX] = self.c4_pool.HOLD_TOKEN_MEMINDEX
        if self.c128_allocator is not None:
            self.c128_allocator.free_all()
            self.full_to_c128_indexs.fill_(-1)
            self.full_to_c128_indexs[self.HOLD_TOKEN_MEMINDEX] = self.c128_pool.HOLD_TOKEN_MEMINDEX
        return

    def alloc_c4(self, need_size) -> torch.Tensor:
        raise AssertionError("DeepSeek-V4 c4 uses page-safe allocation; call alloc_c4_pages instead")

    def alloc_c128(self, need_size) -> torch.Tensor:
        return self.c128_allocator.alloc(need_size)

    def free_c4(self, free_index) -> None:
        raise AssertionError("DeepSeek-V4 c4 uses page live-count release; call evict_c4 instead")

    def free_c128(self, free_index) -> None:
        self.c128_allocator.free(free_index)

    # ------------------------------------------------------------------ packed codecs (torch reference)
    # 与 sglang/vllm 的 fp8_ds_mla 字节布局逐位对齐(ue8m0 幂次 scale)。这些 torch 实现是该 ABI 的
    # 可执行规格(单测 oracle，triton writer 与其逐字节对拍)，不可删除。
    def _pack_mla_kv(self, kv: torch.Tensor) -> torch.Tensor:
        kv = kv.reshape(-1, self.mla_head_dim)
        out = torch.empty((kv.shape[0], self.mla_bytes_per_token), dtype=torch.uint8, device=kv.device)
        nope = kv[:, : self.mla_nope_dim].float().reshape(-1, self.mla_scale_bytes - 1, self.mla_quant_group_size)
        scale = torch.clamp(nope.abs().amax(dim=-1) / DSV4_FP8_E4M3_MAX, min=DSV4_FP8_SCALE_MIN)
        scale_exp = torch.ceil(torch.log2(scale)).to(torch.int32)
        scale = torch.exp2(scale_exp.float())
        nope_fp8 = torch.clamp(nope / scale.unsqueeze(-1), -DSV4_FP8_E4M3_MAX, DSV4_FP8_E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        out[:, : self.mla_nope_dim].copy_(nope_fp8.reshape(-1, self.mla_nope_dim).view(dtype=torch.uint8))
        rope_start = self.mla_nope_dim
        rope_end = rope_start + self.mla_rope_dim * 2
        rope = kv[:, self.mla_nope_dim : self.mla_head_dim].contiguous().to(torch.bfloat16)
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
        scale = torch.clamp(
            k_float.abs().amax(dim=-1, keepdim=True) / DSV4_FP8_E4M3_MAX,
            min=DSV4_FP8_SCALE_MIN,
        )
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
    def pack_mla_kv_to_cache(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        """标准 operator 写入路径。要求本步已对 mem_index 调过 ``alloc_swa``(prep 阶段)；
        HOLD/padding 槽位映射到 swa HOLD 槽，写入无害。"""
        if kv.shape[0] == 0:
            return
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_kv_flashmla_dsv4 import (
            destindex_copy_kv_flashmla_dsv4,
        )

        swa_slots = self.full_to_swa_indexs[mem_index.cuda().long().reshape(-1)]
        destindex_copy_kv_flashmla_dsv4(
            kv.reshape(-1, self.mla_head_dim),
            swa_slots,
            self.swa_pool.get_layer_buffer(layer_index),
            self.swa_pool.page_size,
        )
        return

    def pack_mla_kv_to_cache_fused_norm_rope(
        self,
        layer_index: int,
        mem_index: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ):
        """同 pack_mla_kv_to_cache，但 rmsnorm + 尾部交错 rope 融合进写入 kernel
        并省掉 bf16 kv 中间量。kv 为 wkv 投影原始输出 [T, head_dim+rope_dim]。"""
        if kv.shape[0] == 0:
            return
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope_cuda import (
            fused_k_norm_rope_flashmla,
        )

        swa_slots = self.full_to_swa_indexs[mem_index.cuda().long().reshape(-1)]
        swa_slots = torch.where(swa_slots < 0, torch.full_like(swa_slots, self.swa_pool.HOLD_TOKEN_MEMINDEX), swa_slots)
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

    def pack_indexer_k_to_cache(self, layer_index: int, slots: torch.Tensor, indexer_k: torch.Tensor):
        if indexer_k.shape[0] == 0:
            return
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 indexer-K"
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import (
            destindex_copy_indexer_k_dsv4,
        )

        destindex_copy_indexer_k_dsv4(
            indexer_k.reshape(-1, self.indexer_head_dim),
            slots.to(indexer_k.device),
            self.c4_indexer_pool.get_layer_buffer(self.layer_to_c4_idx[layer_index]),
            self.c4_indexer_pool.page_size,
        )

    def gather_indexer_k(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        """反量化 gather c4 indexer-K: slots [N](c4 槽位,HOLD 合法) -> [N, indexer_head_dim] bf16。
        indexer top-k 打分用(纯张量操作,cuda-graph 安全)。"""
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 indexer-K"
        pool = self.c4_indexer_pool
        flat = pool.get_layer_buffer(self.layer_to_c4_idx[layer_index]).view(-1)
        data_offsets, scale_offsets = pool._loc_offsets(slots.reshape(-1))
        data_range = torch.arange(pool.data_bytes_per_token, device=flat.device)
        scale_range = torch.arange(pool.scale_bytes_per_token, device=flat.device)
        k_fp8 = flat[data_offsets.unsqueeze(1) + data_range.unsqueeze(0)].view(torch.float8_e4m3fn)
        scale = flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)].contiguous().view(torch.float32)
        return (k_fp8.float() * scale).to(torch.bfloat16)

    # ------------------------------------------------------------------ fenced inherited APIs
    # kv_buffer 是 page 索引的 uint8 slab，基类按 token 索引读写的接口会静默写坏数据，显式拦截。
    def get_index_kv_buffer(self, index):
        raise NotImplementedError("DeepSeek-V4 packed page-slab cache does not support token-indexed kv_buffer io")

    def load_index_kv_buffer(self, index, load_tensor_dict):
        raise NotImplementedError("DeepSeek-V4 packed page-slab cache does not support token-indexed kv_buffer io")

    def alloc_kv_move_buffer(self, max_req_total_len):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def write_mem_to_page_kv_move_buffer(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def read_page_kv_move_buffer_to_mem(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def send_to_decode_node(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def receive_from_prefill_node(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def send_to_decode_node_p2p(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def receive_from_prefill_node_p2p(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")
