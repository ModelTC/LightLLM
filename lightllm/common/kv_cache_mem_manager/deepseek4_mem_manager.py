import torch
import triton
from dataclasses import dataclass
from typing import List, Optional
from .mem_manager import MemoryManager
from .operator import DeepseekV4MemOperator
from .allocator import KvCacheAllocator
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.log_utils import init_logger

from .deepseek4_cache_layout import (
    DSV4_MLA_SCALE_BYTES,
    DSV4_MLA_DATA_BYTES_PER_TOKEN,
    DSV4_SWA_PAGE_SIZE,
    DSV4_CACHE_BLOCK_SIZE,
    DeepseekV4CpuCacheLayout,
    DeepseekV4PDCacheLayout,
    packed_page_nbytes,
)

logger = init_logger(__name__)


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
    """Packed cache storage: token data rows followed by per-token scales.

    The pool owns buffers and byte strides. Memory/request managers allocate
    logical slots; Triton writers consume layer buffers directly.
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
        hold_size: int = 1,
    ):
        self.size = size
        self.page_size = page_size
        self.layer_num = layer_num
        self.data_bytes_per_token = data_bytes
        self.scale_bytes_per_token = scale_bytes
        self.bytes_per_token = data_bytes + scale_bytes
        self.num_pages = triton.cdiv(size + hold_size, page_size)
        self.bytes_per_page = packed_page_nbytes(page_size, data_bytes, scale_bytes, align_bytes)
        self.scale_offset_in_page = page_size * data_bytes
        self.buffer = torch.zeros((layer_num, self.num_pages, self.bytes_per_page), dtype=torch.uint8, device=device)
        self.HOLD_TOKEN_MEMINDEX = size

    def get_layer_buffer(self, layer_index: int) -> torch.Tensor:
        return self.buffer[layer_index]


class DeepseekV4MemoryManager(MemoryManager):
    """DeepSeek-V4 KV cache: 窗口 latent(全层) + c4/c128 压缩 latent(压实层) + c4 indexer-K。

    与兄弟 manager 一致的 token-slot 设计；req 索引的表都在 DeepseekV4ReqManager。

    - ``swa_pool``: packed latent for all layers. The request manager owns its
      physical pages and request-private page table. Prefill retains the entire
      current chunk, without a fixed-size ring that could overwrite unread KV.
    - ``c4_pool``/``c128_pool``: 压缩 latent，按 qwen3next 的层号压实手法只为压缩层建层；
      c4 另带 packed indexer-K 池。每个 256-token block 对应三个 packed 历史页；
      压缩槽位直接由组末 full slot 除以压缩率得到，随 full 页分配和释放。
    - 写入走模型专用的 fused norm/RoPE packed writer，显式接收本轮 SWA 槽；
    """

    operator_class = DeepseekV4MemOperator

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
        swa_page_num: int,
        indexer_head_dim: int = 128,
        cpu_cache_token_page_size: int = 2048,
        always_copy=False,
        mem_fraction=0.9,
    ):
        assert head_num == 1, "DeepSeek-V4 是 MLA(MQA)，dense latent 的 head_num 必须为 1"
        assert len(compress_rates) == layer_num, f"compress_rates 长度 {len(compress_rates)} 必须等于 layer_num {layer_num}"
        assert all(r in (0, 4, 128) for r in compress_rates), "compress_rates 取值只能是 0/4/128"
        assert 0 <= mtp_step < 128, "mtp_step 必须位于 [0, 128)"

        self.compress_rates = list(compress_rates)
        self.n_c4 = sum(1 for r in self.compress_rates if r == 4)
        self.n_c128 = sum(1 for r in self.compress_rates if r == 128)
        self.indexer_head_dim = indexer_head_dim
        self.max_request_num = max_request_num
        # C4 retains two overlapping groups; both rings also hold speculative candidates.
        self.c4_state_ring = 2 * 4 + mtp_step
        self.c128_state_ring = triton.cdiv(128 + mtp_step, 4) * 4
        self.swa_num_pages = int(swa_page_num)
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

        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    # ------------------------------------------------------------------ sizing
    @staticmethod
    def _paged_state_rows(num_swa_pages: int, ring: int, ratio: int) -> int:
        rows = num_swa_pages * ring + ring + 1
        return triton.cdiv(rows, ratio) * ratio

    @staticmethod
    def _init_state_sentinel(buffer: torch.Tensor) -> None:
        half = buffer.shape[-1] // 2
        buffer[:, -1, :half].zero_()
        buffer[:, -1, half:].fill_(float("-inf"))
        return

    def get_cell_size(self):
        # Profile the physical PackedPagePool stride, not 584 logical bytes per MLA slot.
        # Its MLA pages round (slots * (576 data + 8 scale)) up to a 576-byte
        # data-row boundary; the indexer pool uses align_bytes=1 (no padding).
        # One 256-token full page owns one C4/indexer page and one C128 page
        # per corresponding layer, so divide their total bytes by 256.
        layout = self.cpu_cache_layout
        c128_page = packed_page_nbytes(
            DSV4_CACHE_BLOCK_SIZE // 128,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
            DSV4_MLA_SCALE_BYTES,
            DSV4_MLA_DATA_BYTES_PER_TOKEN,
        )
        return (
            self.n_c4 * (layout.c4_gpu_page_nbytes + layout.c4_indexer_gpu_page_nbytes) + self.n_c128 * c128_page
        ) / DSV4_CACHE_BLOCK_SIZE

    def get_fixed_memory_size(self):
        layout = self.cpu_cache_layout
        swa_bytes = self.layer_num * (self.swa_num_pages + 1) * layout.swa_gpu_page_nbytes  # includes the HOLD page
        c4_rows = self._paged_state_rows(self.swa_num_pages, self.c4_state_ring, 4)
        c4_state_bytes = self.n_c4 * c4_rows * (layout.c4_state_row_nbytes + layout.c4_indexer_state_row_nbytes)
        c128_rows = (self.max_request_num + 1) * self.c128_state_ring + 1
        c128_state_bytes = self.n_c128 * c128_rows * (2 * self.head_dim) * torch.float32.itemsize
        compressed_hold_bytes = int(self.get_cell_size() * get_env_start_args().page_size)
        return swa_bytes + c4_state_bytes + c128_state_bytes + compressed_hold_bytes

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

        self.swa_size = self.swa_num_pages * DSV4_SWA_PAGE_SIZE
        self.swa_pool = PackedPagePool(
            size=self.swa_size,
            page_size=DSV4_SWA_PAGE_SIZE,
            layer_num=layer_num,
            data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
            scale_bytes=DSV4_MLA_SCALE_BYTES,
            align_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
        )
        # 注意: 该别名是 page 索引([layer, num_pages, bytes_per_page])而非 token 索引，
        # 只允许 get_att_input_params 的消费者使用；token 索引语义的继承接口已显式 fence。
        self.kv_buffer = self.swa_pool.buffer
        # 页粒度分配(页 = 128 槽,位置对齐): 槽位不变式 slot(p) = page_base + p%128。
        # swa_size 整页对齐 ⇒ HOLD 槽(swa_size)独占池子最后一个物理页,永不参与分配。
        self.swa_page_allocator = KvCacheAllocator(
            self.swa_num_pages, shared_name=f"{server}_dsv4_swa_can_use_page_num_{rank_in_node}"
        )
        self.HOLD_TOKEN_MEMINDEX = size

        self.c4_size = triton.cdiv(size, 4)
        self.c128_size = triton.cdiv(size, 128)
        self.c4_pool: Optional[PackedPagePool] = None
        self.c4_indexer_pool: Optional[PackedPagePool] = None
        self.c128_pool: Optional[PackedPagePool] = None
        self.c4_state_buffer: Optional[torch.Tensor] = None
        self.c4_indexer_state_buffer: Optional[torch.Tensor] = None
        self.c128_state_buffer: Optional[torch.Tensor] = None
        if self.n_c4 > 0:
            self.c4_pool = PackedPagePool(
                size=self.c4_size,
                page_size=DSV4_CACHE_BLOCK_SIZE // 4,
                layer_num=self.n_c4,
                data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                scale_bytes=DSV4_MLA_SCALE_BYTES,
                align_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                hold_size=self.page_size // 4,
            )
            self.c4_indexer_pool = PackedPagePool(
                size=self.c4_size,
                page_size=DSV4_CACHE_BLOCK_SIZE // 4,
                layer_num=self.n_c4,
                data_bytes=self.indexer_head_dim,
                scale_bytes=torch.float32.itemsize,
                hold_size=self.page_size // 4,
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
                page_size=DSV4_CACHE_BLOCK_SIZE // 128,
                layer_num=self.n_c128,
                data_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                scale_bytes=DSV4_MLA_SCALE_BYTES,
                align_bytes=DSV4_MLA_DATA_BYTES_PER_TOKEN,
                hold_size=self.page_size // 128,
            )
            # c128 compressor 在途状态按 request 寻址。每个 request 保留完整的 128-token
            # 聚合窗口以及 MTP 候选余量；最后一行供无效请求/位置读取哨兵。
            state_rows = (self.max_request_num + 1) * self.c128_state_ring + 1
            self.c128_state_buffer = torch.zeros(
                (self.n_c128, state_rows, 2 * self.head_dim), dtype=torch.float32, device="cuda"
            )
            self._init_state_sentinel(self.c128_state_buffer)

        from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4StateCacheManager

        self.state_cache_shapes = dict(
            swa_shape=(layer_num, DSV4_CACHE_BLOCK_SIZE // self.swa_pool.page_size, self.swa_pool.bytes_per_page),
            c4_shape=(self.n_c4, 4, 4 * head_dim),
            indexer_shape=(self.n_c4, 4, 4 * self.indexer_head_dim),
        )
        args = get_env_start_args()
        big_page_tokens = args.linear_att_hash_page_size * args.linear_att_page_block_num
        self.big_page_buffers = DeepseekV4StateCacheManager(
            size=triton.cdiv(size, big_page_tokens),
            **self.state_cache_shapes,
        )

        logger.info(
            f"DeepseekV4MemoryManager pools: full_tokens={size} swa={self.swa_size}({self.swa_num_pages}p) "
            f"c4={self.c4_size}(L={self.n_c4}) c128={self.c128_size}(L={self.n_c128}) "
            f"c4_state_ring={self.c4_state_ring} c128_state_ring={self.c128_state_ring} "
            f"max_requests={self.max_request_num} "
            f"packed_kv_bytes={self.swa_pool.bytes_per_token} "
            f"indexer_bytes={self.indexer_head_dim + torch.float32.itemsize}"
        )

    # ------------------------------------------------------------------ buffer accessors
    def get_att_input_params(self, layer_index: int):
        return self.swa_pool.get_layer_buffer(layer_index)

    def get_compressed_kv_buffer(self, layer_index: int) -> torch.Tensor:
        if self.compress_rates[layer_index] == 4:
            return self.c4_pool.get_layer_buffer(self.layer_to_c4_idx[layer_index])
        return self.c128_pool.get_layer_buffer(self.layer_to_c128_idx[layer_index])

    def get_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        return self.c4_indexer_pool.get_layer_buffer(self.layer_to_c4_idx[layer_index])

    def get_c4_state_buffer(self, layer_index: int) -> torch.Tensor:
        return self.c4_state_buffer[self.layer_to_c4_idx[layer_index]]

    def get_c4_indexer_state_buffer(self, layer_index: int) -> torch.Tensor:
        return self.c4_indexer_state_buffer[self.layer_to_c4_idx[layer_index]]

    def get_c128_state_buffer(self, layer_index: int) -> torch.Tensor:
        return self.c128_state_buffer[self.layer_to_c128_idx[layer_index]]

    # ------------------------------------------------------------------ CPU cache load resources
    def prepare_cpu_cache_load(
        self,
        token_num: int,
        loaded_end: int,
        resume_swa_slots: torch.Tensor,
        mem_indexes: Optional[torch.Tensor] = None,
    ) -> DeepseekV4CpuCacheLoadPlan:
        """Prepare a missing history suffix using the request's continuation slots.

        ``loaded_end`` is the CPU checkpoint boundary.  ``token_num`` may be
        smaller than the checkpoint page when a GPU radix prefix overlaps its
        beginning, but both endpoints remain 256-token aligned.
        The common CPU-cache path supplies reserved slots; direct callers may
        request their allocation here.
        """
        token_num = int(token_num)
        loaded_end = int(loaded_end)
        layout = self.cpu_cache_layout
        if token_num <= 0 or token_num % DSV4_CACHE_BLOCK_SIZE != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache load size must be a positive multiple of "
                f"{DSV4_CACHE_BLOCK_SIZE}, got {token_num}"
            )
        if loaded_end < token_num or loaded_end % layout.token_page_size != 0:
            raise ValueError(
                f"DeepSeek-V4 CPU cache loaded_end must be a checkpoint boundary >= token_num, "
                f"got loaded_end={loaded_end}, token_num={token_num}, page={layout.token_page_size}"
            )

        block_num = token_num // DSV4_CACHE_BLOCK_SIZE
        device = self.swa_pool.buffer.device
        if mem_indexes is None:
            mem_indexes = self.alloc(token_num).to(device, non_blocking=True)
        history_full_slots = mem_indexes.view(block_num, DSV4_CACHE_BLOCK_SIZE)
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
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import (
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

    def pack_indexer_k_to_cache(
        self,
        layer_index: int,
        mem_index: torch.Tensor,
        positions: torch.Tensor,
        indexer_k: torch.Tensor,
    ):
        if indexer_k.shape[0] == 0:
            return
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
