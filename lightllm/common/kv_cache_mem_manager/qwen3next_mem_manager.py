import torch
import triton
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.common.linear_att_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from .operator import LinearAttMemOperator
from typing import Tuple, Any, List

logger = init_logger(__name__)


class Qwen3NextMemManager(MemoryManager):
    operator_class = LinearAttMemOperator

    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        full_att_layer_num,
        linear_config: LinearAttCacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.linear_config = linear_config

        super().__init__(size, dtype, num_kv_heads, head_dim, full_att_layer_num, always_copy, mem_fraction)

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = layer_index // self.linear_config.full_attention_interval
        return super().get_att_input_params(layer_index)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # TODO 初始化线性 att 对应的部分 buffer.
        self._init_linear_att_buffers()
        return

    def _init_linear_att_buffers(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        # 申请大页可能需要对应的资源, 多申请了两个linear att的状态，理论上这个状态
        # 永远不会被 alloc 申请到，只会在 cpu cache中，用于过渡和存储碎页情况下的
        # cpu cache 的页面拷贝。
        self.linear_att_big_page_buffers = LinearAttCacheManager(
            size=triton.cdiv(self.size, big_page_token_num) + 2,
            linear_config=self.linear_config,
            keep_num=2,
        )

        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 1
        return

    def _free_buffers(self):
        super()._free_buffers()
        self._free_linear_att_buffers()
        return

    def _free_linear_att_buffers(self):
        self.linear_att_big_page_buffers = None
        return

    def write_to_shm(self, req_manager):
        self.req_to_conv_state = req_manager.req_to_conv_state
        self.req_to_ssm_state = req_manager.req_to_ssm_state
        return super().write_to_shm(req_manager)

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        kv_move_buffer = super().alloc_paged_kv_move_buffer(page_num, page_size)
        Qwen3NextLinearAttPageHelper(self).assert_page_size()
        return kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes,
        page_index: int,
        dp_index: int,
        mem_managers,
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        if page_kind == "kv":
            return super().write_mem_to_page_kv_move_buffer(
                mem_indexes=mem_indexes,
                page_index=page_index,
                dp_index=dp_index,
                mem_managers=mem_managers,
                dp_world_size=dp_world_size,
                page_kind=page_kind,
                req_idx=req_idx,
            )
        assert page_kind == "linear_att_state", f"unknown page_kind={page_kind}"
        assert req_idx is not None
        helper = Qwen3NextLinearAttPageHelper(self)
        dp_mems = helper.get_dp_mems(mem_managers, dp_index, dp_world_size)
        helper.write_req_to_page(page_index=page_index, req_idx=req_idx, dp_mems=dp_mems)
        return

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes,
        page_index: int,
        dp_index: int,
        mem_managers,
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        if page_kind == "kv":
            return super().read_page_kv_move_buffer_to_mem(
                mem_indexes=mem_indexes,
                page_index=page_index,
                dp_index=dp_index,
                mem_managers=mem_managers,
                dp_world_size=dp_world_size,
                page_kind=page_kind,
                req_idx=req_idx,
            )
        assert page_kind == "linear_att_state", f"unknown page_kind={page_kind}"
        assert req_idx is not None
        helper = Qwen3NextLinearAttPageHelper(self)
        dp_mems = helper.get_dp_mems(mem_managers, dp_index, dp_world_size)
        helper.read_page_to_req(page_index=page_index, req_idx=req_idx, dp_mems=dp_mems)
        return


class Qwen3NextLinearAttPageHelper:
    def __init__(self, mem_manager: "Qwen3NextMemManager"):
        self.mem_manager = mem_manager
        self.linear_config = mem_manager.linear_config
        self.req_to_conv_state = mem_manager.req_to_conv_state
        self.req_to_ssm_state = mem_manager.req_to_ssm_state
        self.global_linear_k_heads = self.linear_config.global_linear_k_heads
        self.global_linear_v_heads = self.linear_config.global_linear_v_heads

        self.global_q_dim = self.global_linear_k_heads * self.linear_config.head_linear_k_dim
        self.global_k_dim = self.global_q_dim
        self.global_v_heads = self.global_linear_v_heads
        self.global_v_dim = self.global_v_heads * self.linear_config.head_linear_v_dim
        # conv state follows mixed_qkv layout: [q, k, v], each as a flat channel block.
        self.conv_shape = (
            self.linear_config.linear_layer_num,
            self.global_q_dim + self.global_k_dim + self.global_v_dim,
            self.linear_config.conv_kernel_size - 1,
        )
        self.ssm_shape = (
            self.linear_config.linear_layer_num,
            self.global_v_heads,
            self.linear_config.head_linear_k_dim,
            self.linear_config.head_linear_v_dim,
        )

        self.conv_nbytes = (
            self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2] * self.req_to_conv_state.buffer.element_size()
        )
        ssm_alignment = self.req_to_ssm_state.buffer.element_size()
        # 做一下字节对齐，防止切出来的不对齐，导致一些操作的性能下降。
        self.ssm_offset = ((self.conv_nbytes + ssm_alignment - 1) // ssm_alignment) * ssm_alignment
        self.ssm_nbytes = (
            self.ssm_shape[0]
            * self.ssm_shape[1]
            * self.ssm_shape[2]
            * self.ssm_shape[3]
            * self.req_to_ssm_state.buffer.element_size()
        )
        self.state_nbytes = self.ssm_offset + self.ssm_nbytes

    def assert_page_size(self):
        kv_move_buffer = self.mem_manager.kv_move_buffer
        page_nbytes = kv_move_buffer[0].numel() * kv_move_buffer.element_size()
        assert (
            page_nbytes >= self.state_nbytes
        ), f"nixl kv move page bytes {page_nbytes} is smaller than global linear att state bytes {self.state_nbytes}"
        return

    def get_dp_mems(self, mem_managers: List["Qwen3NextMemManager"], dp_index: int, dp_world_size: int):
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        assert len(dp_mems) == dp_world_size
        for mem in dp_mems:
            assert hasattr(mem, "req_to_conv_state") and hasattr(mem, "req_to_ssm_state")
            assert mem.linear_config.linear_layer_num == self.linear_config.linear_layer_num
            assert mem.linear_config.conv_kernel_size == self.linear_config.conv_kernel_size
            assert mem.linear_config.head_linear_k_dim == self.linear_config.head_linear_k_dim
            assert mem.linear_config.head_linear_v_dim == self.linear_config.head_linear_v_dim
            assert mem.linear_config.num_linear_k_heads == self.linear_config.num_linear_k_heads
            assert mem.linear_config.num_linear_v_heads == self.linear_config.num_linear_v_heads
        return dp_mems

    def view_page_to_linear_att_state(self, page_index: int):
        page_bytes = self.mem_manager.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        conv_page = page_bytes[0 : self.conv_nbytes].view(self.req_to_conv_state.buffer.dtype).view(self.conv_shape)
        ssm_page = (
            page_bytes[self.ssm_offset : self.ssm_offset + self.ssm_nbytes]
            .view(self.req_to_ssm_state.buffer.dtype)
            .view(self.ssm_shape)
        )
        return conv_page, ssm_page

    def write_req_to_page(
        self,
        page_index: int,
        req_idx: int,
        dp_mems: List["Qwen3NextMemManager"],
    ):
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        for tp_index, mem in enumerate(dp_mems):
            self._write_one_rank(mem, tp_index, conv_req_idx, ssm_req_idx, conv_page, ssm_page)
        return

    def read_page_to_req(
        self,
        page_index: int,
        req_idx: int,
        dp_mems: List["Qwen3NextMemManager"],
    ):
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        for tp_index, mem in enumerate(dp_mems):
            self._read_one_rank(mem, tp_index, conv_req_idx, ssm_req_idx, conv_page, ssm_page)
        return

    def _get_req_state_indexes(self, req_idx: int):
        mtp_size = get_env_start_args().mtp_step + 1
        # Conv is one widened slot per request; SSM keeps the historical S+1 block layout.
        return req_idx, req_idx * mtp_size

    def _write_one_rank(
        self,
        mem: "Qwen3NextMemManager",
        tp_index: int,
        conv_req_idx: int,
        ssm_req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_state = mem.req_to_conv_state.buffer[:, conv_req_idx, ..., : self.conv_shape[-1]]
        ssm_state = mem.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_conv_state_to_page(conv_state, conv_page, mem, tp_index)
        self._copy_ssm_state_to_page(ssm_state, ssm_page, mem, tp_index)
        return

    def _copy_conv_state_to_page(
        self,
        conv_state: torch.Tensor,
        conv_page: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        local_q_heads = mem.linear_config.num_linear_k_heads
        local_v_heads = mem.linear_config.num_linear_v_heads
        head_k_dim = mem.linear_config.head_linear_k_dim
        head_v_dim = mem.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, mem.linear_config.tp_world_size, is_write=True
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            global_q_page[:, global_dim_start:global_dim_end, :].copy_(
                local_q_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
            global_k_page[:, global_dim_start:global_dim_end, :].copy_(
                local_k_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, mem.linear_config.tp_world_size, is_write=True
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            global_v_page[:, global_dim_start:global_dim_end, :].copy_(
                local_v_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
        return

    def _copy_ssm_state_to_page(
        self,
        ssm_state: torch.Tensor,
        ssm_page: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            mem.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            mem.linear_config.tp_world_size,
            is_write=True,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_page[:, global_head_start:global_head_end, :, :].copy_(
                ssm_state[:, local_head_start:local_head_end, :, :],
                non_blocking=True,
            )
        return

    def _get_head_slice(
        self,
        tp_index: int,
        local_heads: int,
        global_heads: int,
        tp_world_size: int,
        is_write: bool,
    ):
        if local_heads == 0 or global_heads == 0:
            return None
        total_local_heads = local_heads * tp_world_size
        repeat_count = max(1, total_local_heads // global_heads)
        if is_write and repeat_count > 1 and tp_index % repeat_count != 0:
            return None
        unique_tp_index = tp_index // repeat_count
        global_head_start = unique_tp_index * local_heads
        global_head_end = min(global_head_start + local_heads, global_heads)
        local_head_start = 0
        local_head_end = global_head_end - global_head_start
        if local_head_end <= local_head_start:
            return None
        return local_head_start, local_head_end, global_head_start, global_head_end

    def _copy_page_to_conv_state(
        self,
        conv_page: torch.Tensor,
        conv_state: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        local_q_heads = mem.linear_config.num_linear_k_heads
        local_v_heads = mem.linear_config.num_linear_v_heads
        head_k_dim = mem.linear_config.head_linear_k_dim
        head_v_dim = mem.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, mem.linear_config.tp_world_size, is_write=False
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            local_q_state[:, local_dim_start:local_dim_end, :].copy_(
                global_q_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
            local_k_state[:, local_dim_start:local_dim_end, :].copy_(
                global_k_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, mem.linear_config.tp_world_size, is_write=False
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            local_v_state[:, local_dim_start:local_dim_end, :].copy_(
                global_v_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
        return

    def _copy_page_to_ssm_state(
        self,
        ssm_page: torch.Tensor,
        ssm_state: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            mem.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            mem.linear_config.tp_world_size,
            is_write=False,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_state[:, local_head_start:local_head_end, :, :].copy_(
                ssm_page[:, global_head_start:global_head_end, :, :],
                non_blocking=True,
            )
        return

    def _read_one_rank(
        self,
        mem: "Qwen3NextMemManager",
        tp_index: int,
        conv_req_idx: int,
        ssm_req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_state = mem.req_to_conv_state.buffer[:, conv_req_idx, ..., : self.conv_shape[-1]]
        ssm_state = mem.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_page_to_conv_state(conv_page, conv_state, mem, tp_index)
        self._copy_page_to_ssm_state(ssm_page, ssm_state, mem, tp_index)
        return
