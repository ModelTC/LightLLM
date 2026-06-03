import torch
from typing import Dict, List, Optional
from .deepseek2_mem_manager import Deepseek2MemoryManager
from .allocator import KvCacheAllocator
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class _SubKvPool:
    """DeepSeek-V4 压缩分支(c4 / c128)使用的轻量子池。

    一个独立的 KvCacheAllocator + 一块压缩 latent buffer，可选附带一块与 latent 1:1 的
    indexer-K buffer(仅 c4/CSA 层用)。刻意不继承 MemoryManager —— pd/shm/kv_move 等机制
    对压缩池暂不需要，保持最小。布局与主 MLA latent 池一致(每槽多预留 1 行作 padding 哨兵)。
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        indexer_head_dim: int = 0,
        shared_name: Optional[str] = None,
        device: str = "cuda",
    ):
        self.size = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.indexer_head_dim = indexer_head_dim

        self.kv_buffer = torch.empty((layer_num, size + 1, head_num, head_dim), dtype=dtype, device=device)
        if indexer_head_dim > 0:
            self.index_k_buffer = torch.empty((layer_num, size + 1, indexer_head_dim), dtype=dtype, device=device)
        else:
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
        return self.kv_buffer[layer_index]

    def get_index_k_buffer(self, layer_index: int) -> torch.Tensor:
        assert self.index_k_buffer is not None, "this sub pool has no indexer-K buffer"
        return self.index_k_buffer[layer_index]


class DeepseekV4MemoryManager(Deepseek2MemoryManager):
    """DeepSeek-V4 KV 管理(锁定决策: SWA 全历史 + 不分页)。

    - dense/SWA latent: 继承 Deepseek2 的单张量 MLA latent ``kv_buffer``(每 token 一槽，所有层
      共享层轴，head_num==1)。SWA 分支靠 layer_infer 传 ``AttControl(use_sliding_window)`` + attn_sink
      读最近窗口;dense 槽为纯 latent，不挂 indexer-K(与 V3.2 区别)。
    - c4_pool / c128_pool: 两个独立 ``_SubKvPool``(window 粒度，1-token 分配)。c4 池附带 indexer-K。
    - 容量: 用闭式 ``get_cell_size()``(= 每个 dense token 在所有池上的总字节)让基类 ``profile_size``
      直接得到 full_token = dense 池大小，再按 1/4、1/128 派生压缩池大小。
    - compressor 递归状态不在这里，放 DeepseekV4ReqManager(后续步骤)。
    """

    # dense 写入沿用 Deepseek2MemOperator(拆 nope/rope);压缩写入算子随 layer_infer 一并补。
    # operator_class 继承自 Deepseek2MemoryManager(= Deepseek2MemOperator)。

    def __init__(
        self,
        size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        compress_rates: List[int],
        indexer_head_dim: int = 128,
        always_copy=False,
        mem_fraction=0.9,
    ):
        assert head_num == 1, "DeepSeek-V4 是 MLA(MQA)，dense latent 的 head_num 必须为 1"
        assert (
            len(compress_rates) == layer_num
        ), f"compress_rates 长度 {len(compress_rates)} 必须等于 layer_num {layer_num}"
        assert all(r in (0, 4, 128) for r in compress_rates), "compress_rates 取值只能是 0/4/128"

        self.compress_rates = list(compress_rates)
        self.n_c4 = sum(1 for r in self.compress_rates if r == 4)
        self.n_c128 = sum(1 for r in self.compress_rates if r == 128)
        self.indexer_head_dim = indexer_head_dim

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

    def get_cell_size(self):
        # 返回“每个 dense(full) token 在所有池上的总字节”。基类 profile_size 用
        # size = available_bytes / get_cell_size()，于是直接得到 full_token = dense 池大小。
        elem = torch._utils._element_size(self.dtype)
        latent_bytes = self.head_num * self.head_dim * elem  # 每 token 每层 dense latent
        dense = latent_bytes * self.layer_num  # SWA 全历史: 所有层
        c4 = latent_bytes * self.n_c4 / 4  # c4 压缩 latent
        c128 = latent_bytes * self.n_c128 / 128  # c128 压缩 latent
        indexer = self.indexer_head_dim * elem * self.n_c4 / 4  # c4 indexer-K
        return dense + c4 + c128 + indexer

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # dense/SWA latent(继承 Deepseek2: [layer_num, size+1, head_num, head_dim])
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        self._init_compressed_pools(size, dtype, head_num, head_dim)

    def _init_compressed_pools(self, size, dtype, head_num, head_dim):
        rank_in_node = get_current_rank_in_node()
        server = get_unique_server_name()

        self.c4_size = (size + 4 - 1) // 4
        self.c128_size = (size + 128 - 1) // 128

        self.c4_pool: Optional[_SubKvPool] = None
        self.c128_pool: Optional[_SubKvPool] = None
        if self.n_c4 > 0:
            self.c4_pool = _SubKvPool(
                size=self.c4_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.n_c4,
                indexer_head_dim=self.indexer_head_dim,
                shared_name=f"{server}_dsv4_c4_can_use_token_num_{rank_in_node}",
            )
        if self.n_c128 > 0:
            self.c128_pool = _SubKvPool(
                size=self.c128_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.n_c128,
                indexer_head_dim=0,
                shared_name=f"{server}_dsv4_c128_can_use_token_num_{rank_in_node}",
            )

        logger.info(
            f"DeepseekV4MemoryManager pools: dense={size} "
            f"c4={self.c4_size}(L={self.n_c4}) c128={self.c128_size}(L={self.n_c128}) "
            f"indexer_head_dim={self.indexer_head_dim}"
        )

    # dense latent 读取沿用父类 get_att_input_params。

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
        if self.c4_pool is not None:
            self.c4_pool.free_all()
        if self.c128_pool is not None:
            self.c128_pool.free_all()
