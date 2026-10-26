import math
import torch
import torch.distributed as dist
from typing import List, Tuple, Any, Union
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from .allocator import KvCacheAllocator
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.config_utils import get_num_key_value_heads
from .operator import BaseMemManagerOperator, NormalMemOperator

logger = init_logger(__name__)


class MemoryManager:
    operator_class = NormalMemOperator
    kv_layer_mapping = None
    kv_is_replicated = False

    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        self.size = size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        self.dtype = dtype
        # profile the max total token num if the size is None
        self.profile_size(mem_fraction)

        # A physical KV page must never straddle the allocator boundary.  The
        # unused remainder is intentionally dropped so every managed page has
        # a stable ``mem_index // page_size`` id.
        self.page_size = get_env_start_args().page_size
        self.size = self.size // self.page_size * self.page_size

        self.allocator = KvCacheAllocator(self.size)

        self._init_buffers(
            self.size,
            dtype,
            head_num,
            head_dim,
            layer_num,
        )
        self.HOLD_TOKEN_MEMINDEXES = tuple(range(self.size, self.size + self.page_size))

        # 构建对外的操作类接口
        self.operator: BaseMemManagerOperator = self.operator_class(self)

    def set_kv_layer_mapping(self, layer_mapping):
        layer_mapping = tuple(layer_mapping)
        if not layer_mapping or any(index is not None and not 0 <= index < self.layer_num for index in layer_mapping):
            raise ValueError("KV layer mapping must refer to allocated layers")
        self.kv_layer_mapping = layer_mapping

    def get_kv_layer_index(self, layer_index):
        kv_layer_index = layer_index if self.kv_layer_mapping is None else self.kv_layer_mapping[layer_index]
        if kv_layer_index is None:
            raise ValueError(f"layer {layer_index} has no KV storage")
        return kv_layer_index

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = self.get_kv_layer_index(layer_index)
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v

    def get_cell_size(self):
        return 2 * self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def get_transfer_view(self, *, include_page_layout=False):
        from lightllm.common.kv_transfer_resources import KVTransferView

        return KVTransferView.from_mem_manager(self, include_page_layout=include_page_layout)

    def get_paged_kv_move_buffer_shape(self, page_num, page_size):
        num_kv_head = get_num_key_value_heads(get_env_start_args().model_dir)
        return (
            page_num,
            page_size,
            self.layer_num,
            2 * num_kv_head,
            self.head_dim,
        )

    def get_pd_kv_move_buffer_size(self):
        args = get_env_start_args()
        if args.run_mode not in ["prefill", "decode"]:
            return 0
        shape = self.get_paged_kv_move_buffer_shape(
            page_num=args.pd_kv_page_num,
            page_size=args.pd_kv_page_size,
        )
        return math.prod(shape) * torch._utils._element_size(self.dtype)

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        torch.cuda.empty_cache()
        world_size = dist.get_world_size()
        available_memory = get_available_gpu_memory(world_size) - get_total_gpu_memory() * (1 - mem_fraction)
        cell_size = self.get_cell_size()
        pd_kv_move_buffer_size = self.get_pd_kv_move_buffer_size()
        available_memory_bytes = available_memory * 1024**3 - pd_kv_move_buffer_size
        self.size = int(available_memory_bytes / cell_size)
        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device=f"cuda:{get_current_device_id()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()
        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight\n"
            f"{str(pd_kv_move_buffer_size / 1024 ** 2)} MB is reserved for PD KV transfer buffer\n"
            f"{str(cell_size / 1024 ** 2)} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        assert size % self.page_size == 0, f"KV cache size {size} must be a multiple of page_size {self.page_size}"
        # 在初始化 kv_buffer 时，每层额外保留一整页；这部分空间不会被 allocator 对外分配。
        # 这一页用于多 DP、overlap microbatch 等模式下的 padding 请求，
        # 页内索引存储在 HOLD_TOKEN_MEMINDEXES 中，与 req_manager 的 HOLD_REQUEST_ID 作用类似。
        self.kv_buffer = torch.empty(
            (layer_num, size + self.page_size, 2 * head_num, head_dim), dtype=dtype, device="cuda"
        )

    def _free_buffers(self):
        self.kv_buffer = None

    def alloc(self, need_size) -> torch.Tensor:
        assert (
            need_size % self.page_size == 0
        ), f"KV cache allocation size {need_size} must be a multiple of page_size {self.page_size}"
        return self.allocator.alloc(need_size)

    def free(self, free_index: Union[torch.Tensor, List[int]]) -> None:
        self.allocator.free(free_index)

    def free_all(self):
        self.allocator.free_all()

    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size // self.page_size * self.page_size
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num

        self.size = size
        self.allocator.resize(size)
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        self.HOLD_TOKEN_MEMINDEXES = tuple(range(self.size, self.size + self.page_size))
        return

    def get_index_kv_buffer(self, index):
        return {"kv_buffer": self.kv_buffer[:, index]}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        self.kv_buffer[:, index].copy_(load_tensor_dict["kv_buffer"])


class ReadOnlyStaticsMemoryManager:
    """
    读取一些统计信息
    """

    def __init__(self) -> None:
        args = get_env_start_args()
        self.global_world_size = args.tp
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = self.global_world_size // args.dp
        # 兼容多机 dp size=1 纯 tp 模式的情况
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.shared_tp_infos = [
            SharedInt(f"mem_manger_can_use_token_num_{rank_in_node}")
            for rank_in_node in range(0, self.node_world_size, self.dp_world_size)
        ]

    def get_unrefed_token_num(self, dp_rank_in_node: int):
        if self.is_multinode_tp:
            return self.shared_tp_infos[0].get_value()
        return self.shared_tp_infos[dp_rank_in_node].get_value()
