import torch
import torch.distributed as dist
from typing import Dict, List, Optional
from .deepseek2_mem_manager import Deepseek2MemoryManager
from .operator import DeepseekV4MemOperator
from .allocator import KvCacheAllocator
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory

logger = init_logger(__name__)


DSV4_MLA_NOPE_DIM = 448
DSV4_MLA_ROPE_DIM = 64
DSV4_MLA_HEAD_DIM = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM
DSV4_MLA_QUANT_GROUP_SIZE = 64
DSV4_MLA_SCALE_BYTES = DSV4_MLA_NOPE_DIM // DSV4_MLA_QUANT_GROUP_SIZE + 1
DSV4_MLA_BYTES_PER_TOKEN = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM * 2 + DSV4_MLA_SCALE_BYTES
DSV4_INDEXER_HEAD_DIM = 128
DSV4_INDEXER_BYTES_PER_TOKEN = DSV4_INDEXER_HEAD_DIM + 4
DSV4_FP8_E4M3_MAX = 448.0
DSV4_FP8_SCALE_MIN = 1e-4
DSV4_MLA_DATA_BYTES_PER_TOKEN = DSV4_MLA_NOPE_DIM + DSV4_MLA_ROPE_DIM * 2
DSV4_MLA_SCALE_TAIL_BYTES = DSV4_MLA_SCALE_BYTES
DSV4_MLA_PAGE_ALIGN_BYTES = DSV4_MLA_DATA_BYTES_PER_TOKEN
DSV4_SWA_PAGE_SIZE = 128
DSV4_C4_PAGE_SIZE = 64
DSV4_C128_PAGE_SIZE = 2
DSV4_PROFILE_MAX_FULL_TOKENS = 1_500_000


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class _PageSlabMlaPool:
    """SGLang-compatible fp8_ds_mla page-slab storage with token-slot addressing.

    The public loc is still a LightLLM token slot. Internally each page stores all
    576B NoPE+RoPE payloads first and the 8B scale records at the page tail:
    data_offset = page * bytes_per_page + token_in_page * 576
    scale_offset = page * bytes_per_page + page_size * 576 + token_in_page * 8
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        device: str = "cuda",
    ):
        self.size = size
        self.page_size = page_size
        self.layer_num = layer_num
        self.dtype = torch.uint8
        self.data_bytes_per_token = DSV4_MLA_DATA_BYTES_PER_TOKEN
        self.scale_bytes_per_token = DSV4_MLA_SCALE_TAIL_BYTES
        self.bytes_per_token = DSV4_MLA_BYTES_PER_TOKEN
        self.num_pages = _ceil_div(size + 1, page_size)
        self.bytes_per_page = (
            _ceil_div(page_size * self.bytes_per_token, DSV4_MLA_PAGE_ALIGN_BYTES) * DSV4_MLA_PAGE_ALIGN_BYTES
        )
        self.scale_offset_in_page = page_size * self.data_bytes_per_token
        self.kv_buffer = torch.zeros(
            (layer_num, self.num_pages, self.bytes_per_page),
            dtype=torch.uint8,
            device=device,
        )
        self.HOLD_TOKEN_MEMINDEX = size

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
        loc = loc.long()
        packed = packed.reshape(-1, DSV4_MLA_BYTES_PER_TOKEN).contiguous()
        flat = self.kv_buffer[layer_index].view(-1)
        data_offsets, scale_offsets = self._loc_offsets(loc)

        data = packed[:, : self.data_bytes_per_token].contiguous()
        scale = packed[:, self.data_bytes_per_token : self.bytes_per_token].contiguous()
        data_range = torch.arange(self.data_bytes_per_token, device=loc.device)
        scale_range = torch.arange(self.scale_bytes_per_token, device=loc.device)
        flat[data_offsets.unsqueeze(1) + data_range.unsqueeze(0)] = data
        flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)] = scale
        return

    def read(self, layer_index: int, loc: torch.Tensor) -> torch.Tensor:
        loc = loc.long()
        if loc.numel() == 0:
            return torch.empty((0, DSV4_MLA_BYTES_PER_TOKEN), dtype=torch.uint8, device=self.kv_buffer.device)
        flat = self.kv_buffer[layer_index].view(-1)
        data_offsets, scale_offsets = self._loc_offsets(loc)
        data_range = torch.arange(self.data_bytes_per_token, device=loc.device)
        scale_range = torch.arange(self.scale_bytes_per_token, device=loc.device)
        data = flat[data_offsets.unsqueeze(1) + data_range.unsqueeze(0)]
        scale = flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)]
        return torch.cat([data, scale], dim=1).contiguous()

    def get_layer_buffer(self, layer_index: int) -> torch.Tensor:
        return self.kv_buffer[layer_index]


class _PageSlabIndexerPool:
    """C4 indexer-K storage: page tail stores per-token fp32 scales."""

    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        device: str = "cuda",
    ):
        self.size = size
        self.page_size = page_size
        self.layer_num = layer_num
        self.head_dim = DSV4_INDEXER_HEAD_DIM
        self.scale_bytes = 4
        self.bytes_per_token = DSV4_INDEXER_BYTES_PER_TOKEN
        self.num_pages = _ceil_div(size + 1, page_size)
        self.bytes_per_page = page_size * self.bytes_per_token
        self.scale_offset_in_page = page_size * self.head_dim
        self.index_k_buffer = torch.zeros(
            (layer_num, self.num_pages, self.bytes_per_page),
            dtype=torch.uint8,
            device=device,
        )
        self.HOLD_TOKEN_MEMINDEX = size

    def _loc_offsets(self, loc: torch.Tensor):
        loc = loc.long()
        page = torch.div(loc, self.page_size, rounding_mode="floor")
        token = loc % self.page_size
        page_base = page * self.bytes_per_page
        k_offsets = page_base + token * self.head_dim
        scale_offsets = page_base + self.scale_offset_in_page + token * self.scale_bytes
        return k_offsets, scale_offsets

    def write(self, layer_index: int, loc: torch.Tensor, packed: torch.Tensor) -> None:
        if loc.numel() == 0:
            return
        loc = loc.long()
        packed = packed.reshape(-1, self.bytes_per_token).contiguous()
        flat = self.index_k_buffer[layer_index].view(-1)
        k_offsets, scale_offsets = self._loc_offsets(loc)
        k_range = torch.arange(self.head_dim, device=loc.device)
        scale_range = torch.arange(self.scale_bytes, device=loc.device)
        flat[k_offsets.unsqueeze(1) + k_range.unsqueeze(0)] = packed[:, : self.head_dim]
        flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)] = packed[:, self.head_dim :]
        return

    def read(self, layer_index: int, loc: torch.Tensor) -> torch.Tensor:
        loc = loc.long()
        if loc.numel() == 0:
            return torch.empty((0, self.bytes_per_token), dtype=torch.uint8, device=self.index_k_buffer.device)
        flat = self.index_k_buffer[layer_index].view(-1)
        k_offsets, scale_offsets = self._loc_offsets(loc)
        k_range = torch.arange(self.head_dim, device=loc.device)
        scale_range = torch.arange(self.scale_bytes, device=loc.device)
        k = flat[k_offsets.unsqueeze(1) + k_range.unsqueeze(0)]
        scale = flat[scale_offsets.unsqueeze(1) + scale_range.unsqueeze(0)]
        return torch.cat([k, scale], dim=1).contiguous()

    def get_layer_buffer(self, layer_index: int) -> torch.Tensor:
        return self.index_k_buffer[layer_index]


class _SubKvPool:
    """Compressed c4/c128 KV pool with token-slot allocator and page-slab backing."""

    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        with_indexer: bool = False,
        shared_name: Optional[str] = None,
        device: str = "cuda",
    ):
        self.size = size
        self.dtype = torch.uint8
        self.layer_num = layer_num
        self.page_size = page_size
        self.mla_pool = _PageSlabMlaPool(size=size, page_size=page_size, layer_num=layer_num, device=device)
        self.kv_buffer = self.mla_pool.kv_buffer
        if with_indexer:
            self.indexer_pool = _PageSlabIndexerPool(
                size=size,
                page_size=page_size,
                layer_num=layer_num,
                device=device,
            )
            self.index_k_buffer = self.indexer_pool.index_k_buffer
        else:
            self.indexer_pool = None
            self.index_k_buffer = None

        self.allocator = KvCacheAllocator(size, shared_name=shared_name)
        self.HOLD_TOKEN_MEMINDEX = size

    def alloc(self, need_size) -> torch.Tensor:
        return self.allocator.alloc(need_size)

    def free(self, free_index) -> None:
        self.allocator.free(free_index)

    def free_all(self) -> None:
        self.allocator.free_all()

    def get_kv_buffer(self, layer_index: int) -> torch.Tensor:
        return self.mla_pool.get_layer_buffer(layer_index)

    def get_index_k_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.indexer_pool is not None, "this sub pool has no indexer-K buffer"
        return self.indexer_pool.get_layer_buffer(layer_index)

    def write_kv(self, layer_index: int, slots: torch.Tensor, packed: torch.Tensor) -> None:
        self.mla_pool.write(layer_index, slots, packed)

    def read_kv(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        return self.mla_pool.read(layer_index, slots)

    def write_indexer_k(self, layer_index: int, slots: torch.Tensor, packed: torch.Tensor) -> None:
        assert self.indexer_pool is not None
        self.indexer_pool.write(layer_index, slots, packed)

    def read_indexer_k(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        assert self.indexer_pool is not None
        return self.indexer_pool.read(layer_index, slots)


class DeepseekV4MemoryManager(Deepseek2MemoryManager):
    """DeepSeek-V4 token-slot KV 管理(584B packed cache + bf16 workspace)。

    - dense/SWA latent: 主 ``kv_buffer`` 仍是 LightLLM 的 token-slot cache，不分页；物理格式改为
      SGLang/vLLM 的 ``fp8_ds_mla``: 448B NoPE fp8 + 64*2B RoPE bf16 + 7B scale + 1B pad = 584B。
    - c4_pool / c128_pool: 两个独立 ``_SubKvPool``(window 粒度，1-token 分配)，compressed KV 同样
      存 584B packed。c4 池附带 132B/token 的 packed indexer-K。
    - 读取时先用 torch reference dequant/gather 回 bf16 workspace，供现有 vLLM sparse FlashMLA wrapper
      消费；下一步可把这些 pack/dequant helper 替换成 fused/triton 版本。
    - 容量: 用闭式 ``get_cell_size()``(= 每个 dense token 在所有池上的 packed 总字节)让基类
      ``profile_size`` 直接得到 full_token = dense 池大小，再按 1/4、1/128 派生压缩池大小。
    - compressor 递归状态放 DeepseekV4ReqManager。
    """

    operator_class = DeepseekV4MemOperator

    mla_nope_dim = DSV4_MLA_NOPE_DIM
    mla_rope_dim = DSV4_MLA_ROPE_DIM
    mla_head_dim = DSV4_MLA_HEAD_DIM
    mla_quant_group_size = DSV4_MLA_QUANT_GROUP_SIZE
    mla_scale_bytes = DSV4_MLA_SCALE_BYTES
    mla_bytes_per_token = DSV4_MLA_BYTES_PER_TOKEN
    indexer_head_dim_default = DSV4_INDEXER_HEAD_DIM
    indexer_bytes_per_token = DSV4_INDEXER_BYTES_PER_TOKEN

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
        self.prefill_dtype = dtype
        self.cache_dtype = torch.uint8
        self.max_request_num = max_request_num
        self.sliding_window = sliding_window
        self._pending_prefill_swa: Dict[int, Dict[str, torch.Tensor]] = {}

        # 全局层号 -> 各压缩池内的压实层号(同 qwen3next 的层号压实手法)
        self.layer_to_c4_idx: Dict[int, int] = {}
        self.layer_to_c128_idx: Dict[int, int] = {}
        c4 = c128 = 0
        for lid, r in enumerate(self.compress_rates):
            if r == 4:
                self.layer_to_c4_idx[lid] = c4
                c4 += 1
            elif r == 128:
                self.layer_to_c128_idx[lid] = c128
                c128 += 1

        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def _planned_swa_size(self, full_size: int) -> int:
        if self.max_request_num is None or self.sliding_window is None:
            return full_size
        window_cap = max(1, int(self.max_request_num) * int(self.sliding_window))
        return max(1, min(full_size, window_cap))

    def _dense_cell_size(self):
        return self.head_num * self.mla_bytes_per_token * self.layer_num

    def _compressed_cell_size(self):
        latent_bytes = self.head_num * self.mla_bytes_per_token
        c4 = latent_bytes * self.n_c4 / 4
        c128 = latent_bytes * self.n_c128 / 128
        indexer = self.indexer_bytes_per_token * self.n_c4 / 4
        return c4 + c128 + indexer

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        torch.cuda.empty_cache()
        world_size = dist.get_world_size()
        available_memory = get_available_gpu_memory(world_size) - get_total_gpu_memory() * (1 - mem_fraction)
        available_bytes = available_memory * 1024 ** 3
        dense_cell = self._dense_cell_size()
        compressed_cell = self._compressed_cell_size()

        if self.max_request_num is not None and self.sliding_window is not None and compressed_cell > 0:
            swa_cap = max(1, int(self.max_request_num) * int(self.sliding_window))
            full_cell = dense_cell + compressed_cell
            bytes_until_swa_cap = full_cell * swa_cap
            if available_bytes <= bytes_until_swa_cap:
                self.size = max(1, int(available_bytes / full_cell))
            else:
                self.size = max(1, int((available_bytes - dense_cell * swa_cap) / compressed_cell))
        else:
            self.size = max(1, int(available_bytes / (dense_cell + compressed_cell)))

        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device=f"cuda:{get_current_device_id()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()

        if self.size > DSV4_PROFILE_MAX_FULL_TOKENS:
            logger.info(
                f"DeepseekV4MemoryManager cap profiled max_total_token_num from "
                f"{self.size} to {DSV4_PROFILE_MAX_FULL_TOKENS} to keep runtime headroom"
            )
            self.size = DSV4_PROFILE_MAX_FULL_TOKENS

        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight\n"
            f"{str((dense_cell + compressed_cell) / 1024 ** 2)} MB is the conservative size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def get_cell_size(self):
        dense = self._dense_cell_size()
        compressed = self._compressed_cell_size()
        if self.size is None:
            return dense + compressed
        swa_ratio = self._planned_swa_size(self.size) / max(1, self.size)
        return dense * swa_ratio + compressed

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.swa_size = self._planned_swa_size(size)
        self.swa_pool = _PageSlabMlaPool(
            size=self.swa_size,
            page_size=DSV4_SWA_PAGE_SIZE,
            layer_num=layer_num,
            device="cuda",
        )
        self.kv_buffer = self.swa_pool.kv_buffer
        self._init_swa_mapping(size)
        self._init_compressed_pools(size, head_num)

    def _init_swa_mapping(self, size):
        rank_in_node = get_current_rank_in_node()
        server = get_unique_server_name()
        self.swa_allocator = KvCacheAllocator(
            self.swa_size,
            shared_name=f"{server}_dsv4_swa_can_use_token_num_{rank_in_node}",
        )
        self.full_to_swa_indexs = torch.full((size + 1,), -1, dtype=torch.int32, device="cuda")
        self.full_to_swa_indexs[size] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        if self.max_request_num is None or self.sliding_window is None:
            self.req_to_swa_indexs = None
            self.req_to_swa_full_indexs = None
            return

        self.req_to_swa_indexs = torch.full(
            (self.max_request_num + 1, self.sliding_window),
            self.swa_pool.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cuda",
        )
        self.req_to_swa_full_indexs = torch.full(
            (self.max_request_num + 1, self.sliding_window),
            -1,
            dtype=torch.int32,
            device="cuda",
        )

    def _init_compressed_pools(self, size, head_num):
        rank_in_node = get_current_rank_in_node()
        server = get_unique_server_name()

        self.c4_size = (size + 4 - 1) // 4
        self.c128_size = (size + 128 - 1) // 128

        self.c4_pool: Optional[_SubKvPool] = None
        self.c128_pool: Optional[_SubKvPool] = None
        if self.n_c4 > 0:
            self.c4_pool = _SubKvPool(
                size=self.c4_size,
                page_size=DSV4_C4_PAGE_SIZE,
                layer_num=self.n_c4,
                with_indexer=True,
                shared_name=f"{server}_dsv4_c4_can_use_token_num_{rank_in_node}",
            )
        if self.n_c128 > 0:
            self.c128_pool = _SubKvPool(
                size=self.c128_size,
                page_size=DSV4_C128_PAGE_SIZE,
                layer_num=self.n_c128,
                with_indexer=False,
                shared_name=f"{server}_dsv4_c128_can_use_token_num_{rank_in_node}",
            )

        logger.info(
            f"DeepseekV4MemoryManager pools: full_tokens={size} swa={self.swa_size} "
            f"c4={self.c4_size}(L={self.n_c4}) c128={self.c128_size}(L={self.n_c128}) "
            f"packed_kv_bytes={self.mla_bytes_per_token} indexer_bytes={self.indexer_bytes_per_token}"
        )

    def get_att_input_params(self, layer_index: int):
        return self.swa_pool.get_layer_buffer(layer_index)

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
        out[:, self.indexer_head_dim : self.indexer_bytes_per_token].copy_(scale.view(dtype=torch.uint8).reshape(-1, 4))
        return out

    def _unpack_indexer_k(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.reshape(-1, self.indexer_bytes_per_token)
        if packed.shape[0] == 0:
            return torch.empty((0, self.indexer_head_dim), dtype=self.dtype, device=packed.device)
        k_fp8 = packed[:, : self.indexer_head_dim].view(dtype=torch.float8_e4m3fn).float()
        scale = packed[:, self.indexer_head_dim : self.indexer_bytes_per_token].view(dtype=torch.float32)
        return (k_fp8 * scale).to(self.dtype)

    def _identity_swa_slots(self, full_slots: torch.Tensor) -> torch.Tensor:
        full_slots = full_slots.long()
        valid = full_slots != self.HOLD_TOKEN_MEMINDEX
        if valid.any() and int(full_slots[valid].max().item()) >= self.swa_size:
            raise RuntimeError(
                "DeepSeek-V4 SWA cache needs req_idx/positions for full token slots outside the SWA pool"
            )
        swa_slots = torch.where(
            valid,
            full_slots,
            torch.full_like(full_slots, self.swa_pool.HOLD_TOKEN_MEMINDEX),
        )
        if valid.any():
            self.full_to_swa_indexs[full_slots[valid]] = swa_slots[valid].to(torch.int32)
        return swa_slots

    def ensure_swa_slots(self, req_idx: int, positions: torch.Tensor, full_slots: torch.Tensor) -> torch.Tensor:
        full_slots = full_slots.long().reshape(-1)
        if full_slots.numel() == 0:
            return full_slots
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None:
            return self._identity_swa_slots(full_slots)

        positions = positions.long().reshape(-1)
        assert positions.numel() == full_slots.numel()
        req_idx = int(req_idx)
        out = torch.empty_like(full_slots, dtype=torch.long)
        for i, (pos, full) in enumerate(zip(positions.tolist(), full_slots.tolist())):
            if full == self.HOLD_TOKEN_MEMINDEX:
                out[i] = self.swa_pool.HOLD_TOKEN_MEMINDEX
                continue

            ring_pos = pos % self.sliding_window
            old_swa = int(self.req_to_swa_indexs[req_idx, ring_pos].item())
            old_full = int(self.req_to_swa_full_indexs[req_idx, ring_pos].item())
            if old_full == full and old_swa != self.swa_pool.HOLD_TOKEN_MEMINDEX:
                swa = old_swa
            elif old_swa != self.swa_pool.HOLD_TOKEN_MEMINDEX:
                if old_full >= 0:
                    self.full_to_swa_indexs[old_full] = -1
                swa = old_swa
            else:
                swa = int(self.swa_allocator.alloc(1)[0].item())

            self.req_to_swa_indexs[req_idx, ring_pos] = swa
            self.req_to_swa_full_indexs[req_idx, ring_pos] = full
            self.full_to_swa_indexs[full] = swa
            out[i] = swa
        return out

    def prepare_decode_swa_slots(
        self,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        mem_index: torch.Tensor,
    ) -> None:
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None:
            return

        reqs = b_req_idx.detach().cpu().tolist()
        seqs = b_seq_len.detach().cpu().tolist()
        fulls = mem_index.detach().cpu().tolist()
        hold = self.swa_pool.HOLD_TOKEN_MEMINDEX
        for req_idx, seq_len, full in zip(reqs, seqs, fulls):
            req_idx = int(req_idx)
            full = int(full)
            if req_idx == self.max_request_num or full == self.HOLD_TOKEN_MEMINDEX:
                continue
            ring_pos = (int(seq_len) - 1) % int(self.sliding_window)
            old_swa = int(self.req_to_swa_indexs[req_idx, ring_pos].item())
            old_full = int(self.req_to_swa_full_indexs[req_idx, ring_pos].item())
            if old_swa == hold:
                old_swa = int(self.swa_allocator.alloc(1)[0].item())
            if old_full >= 0 and old_full != full:
                self.full_to_swa_indexs[old_full] = -1
            self.req_to_swa_indexs[req_idx, ring_pos] = old_swa
            self.req_to_swa_full_indexs[req_idx, ring_pos] = full
            self.full_to_swa_indexs[full] = old_swa
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = hold
        return

    def _reserve_prefill_swa_slots(
        self,
        req_idx: int,
        positions: torch.Tensor,
        full_slots: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_slots = full_slots.long().reshape(-1)
        positions = positions.long().reshape(-1)
        assert positions.numel() == full_slots.numel()

        out = torch.empty_like(full_slots, dtype=torch.long)
        ring_to_swa: Dict[int, int] = {}
        ring_to_old_full: Dict[int, int] = {}
        ring_to_final_full: Dict[int, int] = {}
        hold = self.swa_pool.HOLD_TOKEN_MEMINDEX

        for i, (pos, full) in enumerate(zip(positions.tolist(), full_slots.tolist())):
            if full == self.HOLD_TOKEN_MEMINDEX:
                out[i] = hold
                continue

            ring_pos = int(pos) % int(self.sliding_window)
            swa = ring_to_swa.get(ring_pos)
            if swa is None:
                old_swa = int(self.req_to_swa_indexs[req_idx, ring_pos].item())
                old_full = int(self.req_to_swa_full_indexs[req_idx, ring_pos].item())
                if old_swa == hold:
                    old_swa = int(self.swa_allocator.alloc(1)[0].item())
                swa = old_swa
                ring_to_swa[ring_pos] = swa
                ring_to_old_full[ring_pos] = old_full

            ring_to_final_full[ring_pos] = int(full)
            out[i] = swa

        rings = sorted(ring_to_final_full)
        return {
            "positions": positions.detach().clone(),
            "full_slots": full_slots.detach().clone(),
            "swa_slots": out.detach().clone(),
            "commit_rings": torch.tensor(rings, dtype=torch.long, device=full_slots.device),
            "commit_full_slots": torch.tensor(
                [ring_to_final_full[r] for r in rings],
                dtype=torch.long,
                device=full_slots.device,
            ),
            "commit_swa_slots": torch.tensor(
                [ring_to_swa[r] for r in rings],
                dtype=torch.long,
                device=full_slots.device,
            ),
            "commit_old_full_slots": torch.tensor(
                [ring_to_old_full[r] for r in rings],
                dtype=torch.long,
                device=full_slots.device,
            ),
        }

    def prepare_prefill_swa_slots(
        self,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        b_start_loc: torch.Tensor,
        mem_index: torch.Tensor,
    ) -> None:
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None:
            return

        self._pending_prefill_swa = {}
        req_list = b_req_idx.detach().cpu().tolist()
        seq_list = b_seq_len.detach().cpu().tolist()
        ready_list = b_ready_cache_len.detach().cpu().tolist()
        start_list = b_start_loc.detach().cpu().tolist()
        for req_idx, seq_len, ready_len, start_loc in zip(req_list, seq_list, ready_list, start_list):
            token_num = int(seq_len) - int(ready_len)
            if token_num <= 0:
                continue
            pos = torch.arange(int(ready_len), int(seq_len), dtype=torch.long, device=mem_index.device)
            slots = mem_index[int(start_loc) : int(start_loc) + token_num]
            self._pending_prefill_swa[int(req_idx)] = self._reserve_prefill_swa_slots(int(req_idx), pos, slots)
        return

    def _get_pending_prefill_swa_slots(
        self,
        req_idx: int,
        positions: torch.Tensor,
        full_slots: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pending = self._pending_prefill_swa.get(int(req_idx))
        if pending is None:
            return None
        if pending["positions"].numel() != positions.numel():
            return None
        if not torch.equal(pending["positions"].to(positions.device), positions.long().reshape(-1)):
            return None
        if not torch.equal(pending["full_slots"].to(full_slots.device), full_slots.long().reshape(-1)):
            return None
        return pending["swa_slots"].to(full_slots.device)

    def commit_prefill_swa_slots(self) -> None:
        if not self._pending_prefill_swa:
            return
        for req_idx, pending in self._pending_prefill_swa.items():
            rings = pending["commit_rings"].to(self.req_to_swa_indexs.device)
            if rings.numel() == 0:
                continue
            old_full = pending["commit_old_full_slots"].to(self.full_to_swa_indexs.device)
            valid_old = old_full >= 0
            if valid_old.any():
                self.full_to_swa_indexs[old_full[valid_old].long()] = -1

            full_slots = pending["commit_full_slots"].to(self.full_to_swa_indexs.device)
            swa_slots = pending["commit_swa_slots"].to(self.full_to_swa_indexs.device)
            self.req_to_swa_indexs[int(req_idx), rings] = swa_slots.to(torch.int32)
            self.req_to_swa_full_indexs[int(req_idx), rings] = full_slots.to(torch.int32)
            self.full_to_swa_indexs[full_slots.long()] = swa_slots.to(torch.int32)
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        self._pending_prefill_swa = {}
        return

    def _swa_slots_from_full(self, full_slots: torch.Tensor) -> torch.Tensor:
        full_slots = full_slots.long().reshape(-1)
        if full_slots.numel() == 0:
            return full_slots
        mapped = self.full_to_swa_indexs[full_slots].long()
        missing = mapped < 0
        if missing.any():
            if self.req_to_swa_indexs is not None:
                bad = int(full_slots[missing][0].item())
                raise RuntimeError(f"DeepSeek-V4 dense KV for full token slot {bad} has been evicted from SWA cache")
            fallback = full_slots[missing]
            fallback_valid = fallback < self.swa_size
            if fallback_valid.all():
                mapped[missing] = fallback
                self.full_to_swa_indexs[fallback] = fallback.to(torch.int32)
            else:
                bad = int(fallback[~fallback_valid][0].item())
                raise RuntimeError(f"DeepSeek-V4 dense KV for full token slot {bad} has been evicted from SWA cache")
        return mapped

    def free_swa_for_req(self, req_idx: int) -> None:
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None:
            return
        req_idx = int(req_idx)
        slots = self.req_to_swa_indexs[req_idx]
        full_slots = self.req_to_swa_full_indexs[req_idx]
        valid_swa = slots != self.swa_pool.HOLD_TOKEN_MEMINDEX
        if valid_swa.any():
            free_slots = torch.unique(slots[valid_swa]).detach().cpu()
            self.swa_allocator.free(free_slots)
        valid_full = full_slots >= 0
        if valid_full.any():
            self.full_to_swa_indexs[full_slots[valid_full].long()] = -1
        self.req_to_swa_indexs[req_idx].fill_(self.swa_pool.HOLD_TOKEN_MEMINDEX)
        self.req_to_swa_full_indexs[req_idx].fill_(-1)
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX

    def snapshot_swa_for_prompt_cache(self, req_idx: int, cache_len: int, full_slots: torch.Tensor):
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None or cache_len <= 0:
            return None
        tail_start = max(0, int(cache_len) - int(self.sliding_window))
        full_slots = full_slots[tail_start:cache_len].long().to(self.kv_buffer.device)
        if full_slots.numel() == 0:
            return None
        swa_slots = self.full_to_swa_indexs[full_slots].long()
        if (swa_slots < 0).any():
            bad = int(full_slots[swa_slots < 0][0].item())
            raise RuntimeError(f"DeepSeek-V4 prompt cache cannot snapshot evicted SWA full slot {bad}")
        return {
            "positions": torch.arange(tail_start, cache_len, dtype=torch.int64, device="cpu"),
            "full_slots": full_slots.detach().cpu(),
            "swa_slots": swa_slots.detach().cpu(),
        }

    def clone_swa_for_prompt_cache(self, req_idx: int, cache_len: int, full_slots: torch.Tensor):
        payload = self.snapshot_swa_for_prompt_cache(req_idx, cache_len, full_slots)
        if payload is None:
            return None

        src_slots = payload["swa_slots"].long().to(self.kv_buffer.device)
        dst_slots = self.swa_allocator.alloc(src_slots.numel()).long().to(self.kv_buffer.device)
        for layer_idx in range(self.layer_num):
            self.swa_pool.write(layer_idx, dst_slots, self.swa_pool.read(layer_idx, src_slots))
        payload["swa_slots"] = dst_slots.detach().cpu()
        return payload

    def detach_swa_for_prompt_cache(self, req_idx: int, swa_payload) -> None:
        if (
            swa_payload is None
            or self.req_to_swa_indexs is None
            or self.req_to_swa_full_indexs is None
            or len(swa_payload["positions"]) == 0
        ):
            return
        req_idx = int(req_idx)
        positions = swa_payload["positions"].tolist()
        full_slots = swa_payload["full_slots"].tolist()
        swa_slots = swa_payload["swa_slots"].tolist()
        for pos, full, swa in zip(positions, full_slots, swa_slots):
            ring_pos = int(pos) % int(self.sliding_window)
            if int(self.req_to_swa_indexs[req_idx, ring_pos].item()) == int(swa) and int(
                self.req_to_swa_full_indexs[req_idx, ring_pos].item()
            ) == int(full):
                self.req_to_swa_indexs[req_idx, ring_pos] = self.swa_pool.HOLD_TOKEN_MEMINDEX
                self.req_to_swa_full_indexs[req_idx, ring_pos] = -1
        return

    def restore_swa_from_prompt_cache(self, swa_payload) -> None:
        if swa_payload is None or len(swa_payload["full_slots"]) == 0:
            return
        full_slots = swa_payload["full_slots"].long().to(self.kv_buffer.device)
        swa_slots = swa_payload["swa_slots"].long().to(self.kv_buffer.device)
        self.full_to_swa_indexs[full_slots] = swa_slots.to(torch.int32)
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        return

    def free_swa_prompt_cache(self, swa_payload) -> None:
        if swa_payload is None or len(swa_payload["swa_slots"]) == 0:
            return
        swa_slots = torch.unique(swa_payload["swa_slots"].long()).detach().cpu()
        self.swa_allocator.free(swa_slots)
        full_slots = swa_payload["full_slots"].long().to(self.kv_buffer.device)
        mapped = self.full_to_swa_indexs[full_slots].long()
        expected = swa_payload["swa_slots"].long().to(self.kv_buffer.device)
        same = mapped == expected
        if same.any():
            self.full_to_swa_indexs[full_slots[same]] = -1
        self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        return

    def _keep_last_swa_writes(self, swa_slots: torch.Tensor, packed: torch.Tensor):
        """Drop duplicate SWA writes generated by long prefill ring reuse."""
        if swa_slots.numel() <= 1:
            return swa_slots, packed

        slots_cpu = swa_slots.detach().cpu().tolist()
        seen = set()
        keep = []
        hold = self.swa_pool.HOLD_TOKEN_MEMINDEX
        for i in range(len(slots_cpu) - 1, -1, -1):
            slot = int(slots_cpu[i])
            if slot == hold or slot in seen:
                continue
            seen.add(slot)
            keep.append(i)
        keep.reverse()
        if len(keep) == len(slots_cpu):
            return swa_slots, packed
        if not keep:
            return swa_slots[:0], packed[:0]
        keep_index = torch.tensor(keep, dtype=torch.long, device=swa_slots.device)
        return swa_slots.index_select(0, keep_index), packed.index_select(0, keep_index)

    def pack_mla_kv_to_cache(
        self,
        layer_index: int,
        mem_index: torch.Tensor,
        kv: torch.Tensor,
        req_idx: Optional[int] = None,
        positions: Optional[torch.Tensor] = None,
    ):
        if kv.shape[0] == 0:
            return
        packed = self._pack_mla_kv(kv)
        if req_idx is None or positions is None:
            swa_slots = self._identity_swa_slots(mem_index).to(kv.device)
        else:
            pending_slots = self._get_pending_prefill_swa_slots(req_idx, positions, mem_index)
            if pending_slots is None:
                swa_slots = self.ensure_swa_slots(req_idx, positions, mem_index).to(kv.device)
            else:
                swa_slots = pending_slots.to(kv.device)
            swa_slots, packed = self._keep_last_swa_writes(swa_slots, packed)
            if swa_slots.numel() == 0:
                return
        self.swa_pool.write(layer_index, swa_slots, packed)

    def pack_decode_mla_kv_to_cache(
        self,
        layer_index: int,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        mem_index: torch.Tensor,
        kv: torch.Tensor,
    ):
        if kv.shape[0] == 0:
            return
        packed = self._pack_mla_kv(kv)
        if self.req_to_swa_indexs is None or self.req_to_swa_full_indexs is None:
            swa_slots = self._identity_swa_slots(mem_index).to(kv.device)
        else:
            req = b_req_idx.long()
            ring = ((b_seq_len.long() - 1) % int(self.sliding_window)).long()
            swa_slots = self.req_to_swa_indexs[req, ring].long()

            old_full = self.req_to_swa_full_indexs[req, ring].long()
            full_slots = mem_index.long()
            old_full = torch.where(old_full >= 0, old_full, full_slots)
            self.full_to_swa_indexs[old_full] = torch.full(
                old_full.shape,
                -1,
                dtype=self.full_to_swa_indexs.dtype,
                device=old_full.device,
            )

            self.req_to_swa_full_indexs[req, ring] = full_slots.to(torch.int32)
            self.full_to_swa_indexs[full_slots] = swa_slots.to(torch.int32)
        self.swa_pool.write(layer_index, swa_slots.to(kv.device), packed)

    def gather_mla_kv_from_swa_slots(self, layer_index: int, swa_slots: torch.Tensor) -> torch.Tensor:
        return self._unpack_mla_kv(self.swa_pool.read(layer_index, swa_slots.to(self.kv_buffer.device)))

    def pack_compressed_kv_to_cache(self, layer_index: int, slots: torch.Tensor, comp: torch.Tensor):
        if comp.shape[0] == 0:
            return
        pool, local_layer = self._pool_and_local_layer(layer_index)
        pool.write_kv(local_layer, slots.to(comp.device), self._pack_mla_kv(comp))

    def pack_c4_indexer_k_to_cache(self, layer_index: int, slots: torch.Tensor, indexer_k: torch.Tensor):
        if indexer_k.shape[0] == 0:
            return
        pool, local_layer = self._pool_and_local_layer(layer_index)
        pool.write_indexer_k(local_layer, slots.to(indexer_k.device), self._pack_indexer_k(indexer_k))

    def gather_mla_kv(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        if slots.numel() == 0:
            return torch.empty((0, self.mla_head_dim), dtype=self.dtype, device=self.kv_buffer.device)
        swa_slots = self._swa_slots_from_full(slots).to(self.kv_buffer.device)
        return self._unpack_mla_kv(self.swa_pool.read(layer_index, swa_slots))

    def gather_compressed_kv(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        if slots.numel() == 0:
            return torch.empty((0, self.mla_head_dim), dtype=self.dtype, device=self.kv_buffer.device)
        pool, local_layer = self._pool_and_local_layer(layer_index)
        return self._unpack_mla_kv(pool.read_kv(local_layer, slots.to(self.kv_buffer.device)))

    def gather_c4_indexer_k(self, layer_index: int, slots: torch.Tensor) -> torch.Tensor:
        if slots.numel() == 0:
            return torch.empty(
                (0, self.indexer_head_dim),
                dtype=self.dtype,
                device=self.kv_buffer.device,
            )
        pool, local_layer = self._pool_and_local_layer(layer_index)
        return self._unpack_indexer_k(pool.read_indexer_k(local_layer, slots.to(self.kv_buffer.device)))

    def _pool_and_local_layer(self, layer_index: int):
        r = self.compress_rates[layer_index]
        if r == 4:
            return self.c4_pool, self.layer_to_c4_idx[layer_index]
        if r == 128:
            return self.c128_pool, self.layer_to_c128_idx[layer_index]
        raise AssertionError(f"layer {layer_index} (rate {r}) 不是压缩层，没有压缩池")

    def get_compressed_kv_buffer(self, layer_index: int) -> torch.Tensor:
        pool, local_layer = self._pool_and_local_layer(layer_index)
        return pool.get_kv_buffer(local_layer)

    def get_compressed_indexer_k_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.compress_rates[layer_index] == 4, "只有 c4(CSA) 层有 indexer-K"
        return self.c4_pool.get_index_k_buffer(self.layer_to_c4_idx[layer_index])

    def alloc_c4(self, need_size) -> torch.Tensor:
        return self.c4_pool.alloc(need_size)

    def alloc_c128(self, need_size) -> torch.Tensor:
        return self.c128_pool.alloc(need_size)

    def free_c4(self, free_index) -> None:
        self.c4_pool.free(free_index)

    def free_c128(self, free_index) -> None:
        self.c128_pool.free(free_index)

    def free_all(self):
        super().free_all()
        if hasattr(self, "swa_allocator"):
            self.swa_allocator.free_all()
        if hasattr(self, "full_to_swa_indexs"):
            self.full_to_swa_indexs.fill_(-1)
            self.full_to_swa_indexs[self.HOLD_TOKEN_MEMINDEX] = self.swa_pool.HOLD_TOKEN_MEMINDEX
        if getattr(self, "req_to_swa_indexs", None) is not None:
            self.req_to_swa_indexs.fill_(self.swa_pool.HOLD_TOKEN_MEMINDEX)
            self.req_to_swa_full_indexs.fill_(-1)
        self._pending_prefill_swa = {}
        if self.c4_pool is not None:
            self.c4_pool.free_all()
        if self.c128_pool is not None:
            self.c128_pool.free_all()

    def alloc_kv_move_buffer(self, max_req_total_len):
        raise NotImplementedError("DeepSeek-V4 packed/composite KV transfer is not implemented")

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        raise NotImplementedError("DeepSeek-V4 packed/composite paged KV transfer is not implemented")
