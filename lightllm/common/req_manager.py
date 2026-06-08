import torch
import collections
from dataclasses import dataclass
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

from lightllm.utils.log_utils import init_logger
from .kv_cache_mem_manager import MemoryManager, DeepseekV4MemoryManager
from typing import List, Optional, TYPE_CHECKING
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import token_id_counter
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import (
    update_req_to_token_id_counter,
)
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.utils.config_utils import get_vocab_size
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.common.linear_att_cache_manager.layer_cache import LayerCache
from lightllm.common.linear_att_cache_manager.linear_att_buffer_manager import (
    LinearAttCacheManager,
)

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq

logger = init_logger(__name__)


@dataclass
class DeepseekV4PromptCachePayload:
    cache_len: int
    c4_slots: Optional[torch.Tensor] = None
    c128_slots: Optional[torch.Tensor] = None
    c4_state: Optional[torch.Tensor] = None
    c4_state_pool: Optional[torch.Tensor] = None
    c4_indexer_state: Optional[torch.Tensor] = None
    c4_indexer_state_pool: Optional[torch.Tensor] = None
    swa: Optional[dict] = None


class DeepseekV4PromptCacheValueOps:
    def __init__(self, req_manager: "DeepseekV4ReqManager"):
        self.req_manager = req_manager

    def slice(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        return self.req_manager.slice_prompt_cache_payload(payload, start, end)

    def concat(self, payloads: List[DeepseekV4PromptCachePayload]):
        return self.req_manager.concat_prompt_cache_payloads(payloads)

    def free(self, payload: DeepseekV4PromptCachePayload):
        self.req_manager.free_prompt_cache_payload(payload)
        return


class _ReqNode:
    def __init__(self, index):
        self.index = index
        self.next: "_ReqNode" = None


class _ReqLinkedList:
    def __init__(self, max_request_num):
        self.nodes = [_ReqNode(i) for i in range(max_request_num)]
        self.marks = [0 for _ in range(max_request_num)]
        self.root_node = _ReqNode(-1)
        for i in range(0, max_request_num - 1):
            self.nodes[i].next = self.nodes[i + 1]
        self.root_node.next = self.nodes[0]
        self.can_alloc_size = max_request_num
        return

    def alloc(self):
        if self.root_node.next is None:
            logger.warning("alloc req index fail")
            return None
        get_node = self.root_node.next
        self.root_node.next = self.root_node.next.next
        assert self.marks[get_node.index] == 0
        self.marks[get_node.index] = 1
        self.can_alloc_size -= 1
        return get_node.index

    def free(self, index):
        assert self.marks[index] == 1
        node = self.nodes[index]
        node.next = self.root_node.next
        self.root_node.next = node
        self.marks[index] = 0
        self.can_alloc_size += 1
        return

    def is_all_free(self):
        return self.can_alloc_size == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqLinkedList(max_request_num)
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.mem_manager = mem_manager
        self.req_sampling_params_manager = ReqSamplingParamsManager(max_request_num)
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int], free_token_index):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)

        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index: int):
        self.req_list.free(free_req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        return

    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)
        return

    def free_all(self):
        self.req_list = _ReqLinkedList(self.max_request_num)
        return


class ReqSamplingParamsManager:
    """
    ReqSamplingParamsManager 将输出采样参数中，确定比较固定的部分，纳入到 gpu buffer中进行管理，这样可以更快捷的
    利用 triton kernel 进行处理，对于那些比较动态(部分处理模式下会动态的修改某些后处理参数)，或者存在特殊处理的后处理参数，
    则保留从 InferSamplingParams 中进行动态读取和动态组batch， 具体使用可以参考
    lightllm/server/router/model_infer/mode_backend/generic_post_process.py 文件中的使用方式。
    """

    def __init__(self, max_request_num):
        # mode ["cpu_counter", "pin_mem_counter", "gpu_counter"]
        self.penalty_counter_mode = get_env_start_args().penalty_counter_mode
        self.vocab_size = get_vocab_size(get_env_start_args().model_dir)
        self.req_to_presence_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_frequency_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_repetition_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_next_token_ids = torch.zeros(
            (max_request_num + 1, 8),
            dtype=torch.int64,
            device="cuda",
        )
        self.req_to_exponential_decay_length_penalty = torch.zeros(
            max_request_num + 1, dtype=torch.float32, device="cuda"
        )

        if self.penalty_counter_mode == "gpu_counter":
            self.req_to_out_token_id_counter = torch.zeros(
                (max_request_num + 1, self.vocab_size), dtype=torch.int32, device="cuda"
            )
        elif self.penalty_counter_mode == "pin_mem_counter":
            self.req_to_out_token_id_counter = torch.zeros(
                (max_request_num + 1, self.vocab_size), dtype=torch.int32, device="cpu", pin_memory=True
            )

    def init_req_sampling_params(self, req: "InferReq"):

        shm_param = req.sampling_param.shm_param
        self.req_to_next_token_ids[req.req_idx][0:1].fill_(req.get_last_gen_token())
        self.req_to_presence_penalty[req.req_idx].fill_(shm_param.presence_penalty)
        self.req_to_frequency_penalty[req.req_idx].fill_(shm_param.frequency_penalty)
        self.req_to_repetition_penalty[req.req_idx].fill_(shm_param.repetition_penalty)
        exponential_decay_length_penalty = shm_param.exponential_decay_length_penalty.to_tuple()
        self.req_to_exponential_decay_length_penalty[req.req_idx].fill_(exponential_decay_length_penalty[1])
        # 提前标记当前请求是否需要统计输出token的计数，因为这个统计可能会导致一些特定场景下后处理效率的下降
        # 所以提前标记不需要进行后处理统计的场景。
        req.need_out_token_id_statistics = not (
            shm_param.presence_penalty == 0.0
            and shm_param.frequency_penalty == 0.0
            and shm_param.repetition_penalty == 1.0
        )

        if self.penalty_counter_mode == "cpu_counter":
            if req.sampling_param.shm_param.input_penalty and req.need_out_token_id_statistics:
                req.out_token_id_count = collections.Counter(req.shm_req.get_prompt_ids())
            else:
                req.out_token_id_count = collections.defaultdict(int)
        else:
            self.req_to_out_token_id_counter[req.req_idx].fill_(0)
            if req.sampling_param.shm_param.input_penalty and req.need_out_token_id_statistics:
                prompt_ids = g_pin_mem_manager.gen_from_list(
                    key="prompt_ids_for_penalty",
                    data=req.shm_req.get_prompt_ids_numpy(),
                    dtype=torch.int32,
                ).cuda(non_blocking=True)
                token_id_counter(
                    prompt_ids=prompt_ids, out_token_id_counter=self.req_to_out_token_id_counter[req.req_idx]
                )
                torch.cuda.current_stream().synchronize()

        return

    def update_reqs_out_token_counter_gpu(
        self, b_req_idx: torch.Tensor, next_token_ids: torch.Tensor, mask: torch.Tensor = None
    ):
        if self.penalty_counter_mode not in ["gpu_counter", "pin_mem_counter"]:
            return

        assert b_req_idx.is_cuda and next_token_ids.is_cuda and b_req_idx.shape[0] == next_token_ids.shape[0]

        update_req_to_token_id_counter(
            b_req_idx=b_req_idx,
            next_token_ids=next_token_ids,
            req_to_out_token_id_counter=self.req_to_out_token_id_counter,
            mask=mask,
        )
        return

    def update_reqs_token_counter(
        self, req_objs: List["InferReq"], next_token_ids: List[int], accept_mark: Optional[List[List[bool]]] = None
    ):
        if self.penalty_counter_mode != "cpu_counter":
            return

        for req_obj, next_token_id in zip(req_objs, next_token_ids):
            if req_obj.need_out_token_id_statistics and req_obj.cur_output_len > 0:
                req_obj.out_token_id_count[next_token_id] += 1
        return

    def gen_cpu_out_token_counter_sampling_params(self, req_objs: List["InferReq"]):
        assert self.penalty_counter_mode == "cpu_counter"

        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_cumsum_seq_len: List[int] = [
            0,
        ]
        cum_sum_len = 0
        for i, req_obj in enumerate(req_objs):
            id_to_count = req_obj.out_token_id_count
            p_token_ids.extend(list(id_to_count.keys()))
            p_token_counts.extend(list(id_to_count.values()))
            cum_sum_len += len(id_to_count)
            p_cumsum_seq_len.append(cum_sum_len)

        p_token_ids_tensor = g_pin_mem_manager.gen_from_list(key="p_token_ids", data=p_token_ids, dtype=torch.int32)
        p_token_counts_tensor = g_pin_mem_manager.gen_from_list(
            key="p_token_counts", data=p_token_counts, dtype=torch.int32
        )
        p_cumsum_seq_len_tensor = g_pin_mem_manager.gen_from_list(
            key="p_cumsum_seq_len", data=p_cumsum_seq_len, dtype=torch.int32
        )

        return (
            p_token_ids_tensor.cuda(non_blocking=True),
            p_token_counts_tensor.cuda(non_blocking=True),
            p_cumsum_seq_len_tensor.cuda(non_blocking=True),
        )


class ReqManagerForMamba(ReqManager):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config: LinearAttCacheConfig):
        super().__init__(max_request_num, max_sequence_length, mem_manager)
        self.mtp_step = get_env_start_args().mtp_step
        self.big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        assert (
            self.mtp_step == 0
        ), "currently only support mtp_step 0 for simplicity, more mtp_step support will be added in the future"
        self.linear_config = linear_config

        self.req_to_conv_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=self.linear_config.conv_state_dtype,
            shape=self.linear_config.get_conv_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        self.req_to_ssm_state = LayerCache(
            size=(max_request_num + 1) * (self.mtp_step + 1),
            dtype=self.linear_config.ssm_state_dtype,
            shape=self.linear_config.get_ssm_state_shape(),
            layer_num=self.linear_config.linear_layer_num,
            device="cuda",
        )
        return

    def init_linear_att_state(self, req: "InferReq"):
        index = req.req_idx * (self.mtp_step + 1)
        conv_state = self.req_to_conv_state.buffer[:, index, ...]
        ssm_state = self.req_to_ssm_state.buffer[:, index, ...]
        conv_state.fill_(0)
        ssm_state.fill_(0)
        return

    def get_mamba_cache(self, layer_idx_in_all: int):
        assert (
            0 <= layer_idx_in_all < self.linear_config.all_layer_num
        ), f"invalid transformer layer index {layer_idx_in_all}"
        layer_idx_in_linear = layer_idx_in_all - (layer_idx_in_all // self.linear_config.full_attention_interval)
        conv_states = self.req_to_conv_state.buffer[layer_idx_in_linear]
        ssm_states = self.req_to_ssm_state.buffer[layer_idx_in_linear]
        return conv_states, ssm_states

    def copy_big_page_buffer_to_linear_att_state(self, big_page_buffer_idx: int, req: "InferReq"):

        from .linear_att_cache_manager import LinearAttCacheManager

        big_page_buffers: LinearAttCacheManager = self.mem_manager.linear_att_big_page_buffers

        conv_state, ssm_state = big_page_buffers.get_state_cache(buffer_idx=big_page_buffer_idx)
        dest_req_idx = req.req_idx * (self.mtp_step + 1)

        self.req_to_conv_state.buffer[:, dest_req_idx, ...] = conv_state
        self.req_to_ssm_state.buffer[:, dest_req_idx, ...] = ssm_state
        return

    def copy_small_page_buffer_to_linear_att_state(
        self, req: "InferReq", linear_att_small_page_buffers: LinearAttCacheManager
    ):
        conv_state, ssm_state = linear_att_small_page_buffers.get_state_cache(
            buffer_idx=req.shared_kv_node.small_page_buffer_idx
        )
        dest_req_idx = req.req_idx * (self.mtp_step + 1)
        # TODO 下面这个从 cpu cache 拷贝数据的 gpu的操作，是否是阻塞的操作。
        # 同时，非连续对象的拷贝，可能存在效率问题。
        self.req_to_conv_state.buffer[:, dest_req_idx, ...] = conv_state
        self.req_to_ssm_state.buffer[:, dest_req_idx, ...] = ssm_state
        return


class DeepseekV4ReqManager(ReqManager):
    """DeepSeek-V4 的请求级管理(锁定决策: SWA 全历史 + 不分页)。

    在基类 ReqManager 之上补三类 V4 专有的 per-request 结构。该对象在 mem manager profile 前创建，
    所以初始化只依赖 config 派生出的 compress_rates/head_dim/indexer_head_dim；真实 mem_manager
    会在 `_init_mem_manager()` 后通过 `bind_mem_manager()` 接入。

      * ``req_to_c4_indexs`` / ``req_to_c128_indexs`` —— (req, 窗口下标) -> 压缩池槽位。
        窗口下标 = position // compress_rate;窗口关闭时由 layer-infer 写入,attention 读取前
        n_windows 列即该 req 的全部压缩条目槽。未填充列为 0(不会被读到,语义同 req_to_token_indexs)。
      * ``req_to_c4_state`` / ``req_to_c128_state`` / ``req_to_c4_indexer_state`` —— compressor 的
        “在途窗口”累加状态(per req、per 压缩层),fp32。形状为
        ``(kv_or_score, coff * ratio, coff * dim)``; c4 因 Ca/Cb overlap 取 ``coff=2``,
        c128 取 ``coff=1``。score 初始化为 ``-inf``，与官方 reference compressor 的
        ``kv_state``/``score_state`` 对齐。
      * entry_count 不另存:= position // compress_rate,可由序列长度推出。
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager: Optional[DeepseekV4MemoryManager] = None,
        compress_rates: Optional[List[int]] = None,
        head_dim: Optional[int] = None,
        indexer_head_dim: Optional[int] = None,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)

        self.mem_manager = mem_manager
        self.compress_rates = list(compress_rates)
        self.n_c4 = sum(1 for r in self.compress_rates if r == 4)
        self.n_c128 = sum(1 for r in self.compress_rates if r == 128)
        self.head_dim = head_dim
        self.indexer_head_dim = indexer_head_dim
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

        # (req, 窗口) -> 压缩槽。列数取 ceil(max_seq / ratio) 留足余量。
        c4_windows = (max_sequence_length + 4 - 1) // 4
        c128_windows = (max_sequence_length + 128 - 1) // 128
        self.req_to_c4_indexs = torch.zeros((max_request_num + 1, c4_windows), dtype=torch.int32, device="cuda")
        self.req_to_c128_indexs = torch.zeros((max_request_num + 1, c128_windows), dtype=torch.int32, device="cuda")
        self._c4_entry_counts = [0 for _ in range(max_request_num + 1)]
        self._c128_entry_counts = [0 for _ in range(max_request_num + 1)]

        # compressor 在途窗口累加状态(fp32): [kv_or_score, coff * ratio, coff * dim].
        state_dtype = torch.float32
        self.req_to_c4_state = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(2, 8, 2 * head_dim),
            layer_num=self.n_c4,
            device="cuda",
        )
        self.req_to_c128_state = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(2, 128, head_dim),
            layer_num=self.n_c128,
            device="cuda",
        )
        self.req_to_c4_indexer_state = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(2, 8, 2 * indexer_head_dim),
            layer_num=self.n_c4,
            device="cuda",
        )
        self.req_to_c4_state_pool = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(1, 8, 4 * head_dim),
            layer_num=self.n_c4,
            device="cuda",
        )
        self.req_to_c128_state_pool = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(1, 128, 2 * head_dim),
            layer_num=self.n_c128,
            device="cuda",
        )
        self.req_to_c4_indexer_state_pool = LayerCache(
            size=max_request_num + 1,
            dtype=state_dtype,
            shape=(1, 8, 4 * indexer_head_dim),
            layer_num=self.n_c4,
            device="cuda",
        )
        self._runtime_states = [{} for _ in range(max_request_num + 1)]
        self._init_all_score_state()
        return

    def bind_mem_manager(self, mem_manager: DeepseekV4MemoryManager):
        assert isinstance(mem_manager, DeepseekV4MemoryManager)
        assert self.compress_rates == mem_manager.compress_rates
        assert self.head_dim == mem_manager.head_dim
        assert self.indexer_head_dim == mem_manager.indexer_head_dim
        self.mem_manager = mem_manager
        return

    def _init_all_score_state(self):
        if self.n_c4 > 0:
            self.req_to_c4_state.buffer[:, :, 1, ...].fill_(float("-inf"))
            self.req_to_c4_indexer_state.buffer[:, :, 1, ...].fill_(float("-inf"))
        if self.n_c128 > 0:
            self.req_to_c128_state.buffer[:, :, 1, ...].fill_(float("-inf"))
        return

    def _reset_compress_cache_req(self, cache: LayerCache, req_idx: int):
        if cache.layer_num == 0:
            return
        cache.buffer[:, req_idx, 0, ...].fill_(0)
        cache.buffer[:, req_idx, 1, ...].fill_(float("-inf"))
        return

    def _reset_state_pool_req(self, cache: LayerCache, req_idx: int):
        if cache.layer_num == 0:
            return
        cache.buffer[:, req_idx, ...].fill_(0)
        return

    def init_compress_state(self, req_idx: int):
        """新请求开始时重置其 compressor 在途状态(对应 mamba 的 init_linear_att_state)。"""
        self.clear_runtime_state(req_idx)
        c4, c128 = self.pop_compress_indices_for_req(req_idx)
        self.free_compress_indices(free_c4_index=c4, free_c128_index=c128)
        if self.n_c4 > 0:
            self._reset_compress_cache_req(self.req_to_c4_state, req_idx)
            self._reset_compress_cache_req(self.req_to_c4_indexer_state, req_idx)
            self._reset_state_pool_req(self.req_to_c4_state_pool, req_idx)
            self._reset_state_pool_req(self.req_to_c4_indexer_state_pool, req_idx)
        if self.n_c128 > 0:
            self._reset_compress_cache_req(self.req_to_c128_state, req_idx)
            self._reset_state_pool_req(self.req_to_c128_state_pool, req_idx)
        return

    def _ensure_compress_slots(self, req_idx: int, ratio: int, entry_start: int, entry_count: int) -> torch.Tensor:
        if entry_count == 0:
            return torch.empty((0,), dtype=torch.int32, device="cuda")
        assert entry_start >= 0 and entry_count >= 0
        assert self.mem_manager is not None, "DeepSeek-V4 mem manager is not bound yet"
        if ratio == 4:
            table = self.req_to_c4_indexs
            counts = self._c4_entry_counts
            alloc = self.mem_manager.alloc_c4
        elif ratio == 128:
            table = self.req_to_c128_indexs
            counts = self._c128_entry_counts
            alloc = self.mem_manager.alloc_c128
        else:
            raise AssertionError(f"invalid DeepSeek-V4 compress ratio {ratio}")

        required_count = entry_start + entry_count
        assert required_count <= table.shape[1], (
            f"DeepSeek-V4 compressed slot table overflow: req={req_idx} "
            f"ratio={ratio} required={required_count} capacity={table.shape[1]}"
        )
        old_count = counts[req_idx]
        if required_count > old_count:
            new_slots_cpu = alloc(required_count - old_count)
            table[req_idx, old_count:required_count] = new_slots_cpu.cuda(non_blocking=True)
            counts[req_idx] = required_count
        return table[req_idx, entry_start:required_count]

    def ensure_c4_slots(self, req_idx: int, entry_start: int, entry_count: int) -> torch.Tensor:
        return self._ensure_compress_slots(req_idx, 4, entry_start, entry_count)

    def ensure_c128_slots(self, req_idx: int, entry_start: int, entry_count: int) -> torch.Tensor:
        return self._ensure_compress_slots(req_idx, 128, entry_start, entry_count)

    def ensure_compress_slots(self, layer_index: int, req_idx: int, entry_start: int, entry_count: int) -> torch.Tensor:
        ratio = self.compress_rates[layer_index]
        if ratio == 4:
            return self.ensure_c4_slots(req_idx, entry_start, entry_count)
        if ratio == 128:
            return self.ensure_c128_slots(req_idx, entry_start, entry_count)
        raise AssertionError(f"layer {layer_index} is not a compressed attention layer")

    def prepare_decode_compress_slots(self, b_req_idx: torch.Tensor, b_seq_len: torch.Tensor) -> None:
        req_list = b_req_idx.detach().cpu().tolist()
        seq_list = b_seq_len.detach().cpu().tolist()
        for req_idx, seq_len in zip(req_list, seq_list):
            req_idx = int(req_idx)
            if req_idx == self.HOLD_REQUEST_ID:
                continue
            seq_len = int(seq_len)
            if self.n_c4 > 0:
                required_c4 = seq_len // 4
                old_c4 = self._c4_entry_counts[req_idx]
                if required_c4 > old_c4:
                    self.ensure_c4_slots(req_idx, old_c4, required_c4 - old_c4)
            if self.n_c128 > 0:
                required_c128 = seq_len // 128
                old_c128 = self._c128_entry_counts[req_idx]
                if required_c128 > old_c128:
                    self.ensure_c128_slots(req_idx, old_c128, required_c128 - old_c128)
        return

    def pop_compress_indices_for_req(self, req_idx: int):
        c4_count = self._c4_entry_counts[req_idx]
        if c4_count > 0:
            c4 = self.req_to_c4_indexs[req_idx, :c4_count].clone()
            self.req_to_c4_indexs[req_idx, :c4_count].fill_(0)
            self._c4_entry_counts[req_idx] = 0
        else:
            c4 = None

        c128_count = self._c128_entry_counts[req_idx]
        if c128_count > 0:
            c128 = self.req_to_c128_indexs[req_idx, :c128_count].clone()
            self.req_to_c128_indexs[req_idx, :c128_count].fill_(0)
            self._c128_entry_counts[req_idx] = 0
        else:
            c128 = None
        return c4, c128

    def free_compress_indices(self, free_c4_index=None, free_c128_index=None):
        if free_c4_index is not None and len(free_c4_index) > 0:
            self.mem_manager.free_c4(free_c4_index)
        if free_c128_index is not None and len(free_c128_index) > 0:
            self.mem_manager.free_c128(free_c128_index)
        return

    def alloc(self):
        req_idx = super().alloc()
        if req_idx is not None:
            self.init_compress_state(req_idx)
        return req_idx

    def clear_runtime_state(self, req_idx: int):
        self._runtime_states[req_idx].clear()
        if self.mem_manager is not None and hasattr(self.mem_manager, "free_swa_for_req"):
            self.mem_manager.free_swa_for_req(req_idx)
        return

    def set_runtime_state(self, req_idx: int, layer_index: int, state: dict):
        self._runtime_states[req_idx][layer_index] = state
        return

    def get_runtime_state(self, req_idx: int, layer_index: int):
        return self._runtime_states[req_idx][layer_index]

    def get_compress_state_for_req(self, layer_index: int, req_idx: int):
        if self.compress_rates[layer_index] == 4:
            state = self.get_c4_compress_state(layer_index)
        elif self.compress_rates[layer_index] == 128:
            state = self.get_c128_compress_state(layer_index)
        else:
            raise AssertionError(f"layer {layer_index} is not a compressed attention layer")
        return state[req_idx, 0], state[req_idx, 1]

    def get_compress_state_pool_for_req(self, layer_index: int, req_idx: int):
        if self.compress_rates[layer_index] == 4:
            cache = self.req_to_c4_state_pool
            local = self.layer_to_c4_idx[layer_index]
        elif self.compress_rates[layer_index] == 128:
            cache = self.req_to_c128_state_pool
            local = self.layer_to_c128_idx[layer_index]
        else:
            raise AssertionError(f"layer {layer_index} is not a compressed attention layer")
        return cache.buffer[local, req_idx]

    def get_c4_compress_state(self, layer_index: int) -> torch.Tensor:
        local = self.layer_to_c4_idx[layer_index]
        return self.req_to_c4_state.buffer[local]

    def get_c128_compress_state(self, layer_index: int) -> torch.Tensor:
        local = self.layer_to_c128_idx[layer_index]
        return self.req_to_c128_state.buffer[local]

    def get_c4_indexer_compress_state(self, layer_index: int) -> torch.Tensor:
        local = self.layer_to_c4_idx[layer_index]
        return self.req_to_c4_indexer_state.buffer[local]

    def get_c4_indexer_state_pool_for_req(self, layer_index: int, req_idx: int) -> torch.Tensor:
        local = self.layer_to_c4_idx[layer_index]
        return self.req_to_c4_indexer_state_pool.buffer[local, req_idx]

    def get_prompt_cache_value_ops(self):
        return DeepseekV4PromptCacheValueOps(self)

    def get_prompt_cache_page_size(self):
        return 128

    def _slice_cpu_slots(self, slots: Optional[torch.Tensor], start: int, end: int, ratio: int):
        if slots is None:
            return None
        return slots[start // ratio : end // ratio].clone()

    def _slice_swa_payload(self, swa_payload, start: int, end: int):
        if swa_payload is None:
            return None
        positions = swa_payload["positions"]
        mask = (positions >= start) & (positions < end)
        if not bool(mask.any()):
            return None
        return {
            "positions": positions[mask].clone(),
            "full_slots": swa_payload["full_slots"][mask].clone(),
            "swa_slots": swa_payload["swa_slots"][mask].clone(),
        }

    def slice_prompt_cache_payload(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        start = int(start)
        end = int(end)
        # c4/c128/indexer-K slots are true historical KV and can be sliced by ratio.
        # compressor running state only describes the payload end boundary; it is valid
        # for a slice only when that slice keeps the original end boundary.
        keep_end_state = end == payload.cache_len
        return DeepseekV4PromptCachePayload(
            cache_len=end - start,
            c4_slots=self._slice_cpu_slots(payload.c4_slots, start, end, 4),
            c128_slots=self._slice_cpu_slots(payload.c128_slots, start, end, 128),
            c4_state=payload.c4_state.clone() if keep_end_state and payload.c4_state is not None else None,
            c4_state_pool=payload.c4_state_pool.clone()
            if keep_end_state and payload.c4_state_pool is not None
            else None,
            c4_indexer_state=payload.c4_indexer_state.clone()
            if keep_end_state and payload.c4_indexer_state is not None
            else None,
            c4_indexer_state_pool=payload.c4_indexer_state_pool.clone()
            if keep_end_state and payload.c4_indexer_state_pool is not None
            else None,
            swa=self._slice_swa_payload(payload.swa, start, end),
        )

    def concat_prompt_cache_payloads(self, payloads: List[DeepseekV4PromptCachePayload]):
        if len(payloads) == 0:
            return None
        c4_slots = [p.c4_slots for p in payloads if p.c4_slots is not None and len(p.c4_slots) > 0]
        c128_slots = [p.c128_slots for p in payloads if p.c128_slots is not None and len(p.c128_slots) > 0]
        last = payloads[-1]
        return DeepseekV4PromptCachePayload(
            cache_len=sum(p.cache_len for p in payloads),
            c4_slots=torch.cat(c4_slots, dim=0) if c4_slots else None,
            c128_slots=torch.cat(c128_slots, dim=0) if c128_slots else None,
            c4_state=last.c4_state,
            c4_state_pool=last.c4_state_pool,
            c4_indexer_state=last.c4_indexer_state,
            c4_indexer_state_pool=last.c4_indexer_state_pool,
            swa=last.swa,
        )

    def build_prompt_cache_payload(
        self,
        req_idx: int,
        cache_len: int,
        clone_swa: bool = False,
    ) -> DeepseekV4PromptCachePayload:
        assert self.mem_manager is not None
        cache_len = int(cache_len)
        full_slots = self.req_to_token_indexs[req_idx, :cache_len].detach().cpu()
        c4_count = cache_len // 4
        c128_count = cache_len // 128
        c4_slots = self.req_to_c4_indexs[req_idx, :c4_count].detach().cpu().clone() if c4_count > 0 else None
        c128_slots = self.req_to_c128_indexs[req_idx, :c128_count].detach().cpu().clone() if c128_count > 0 else None
        if clone_swa:
            swa_payload = self.mem_manager.clone_swa_for_prompt_cache(req_idx, cache_len, full_slots)
        else:
            swa_payload = self.mem_manager.snapshot_swa_for_prompt_cache(req_idx, cache_len, full_slots)
        return DeepseekV4PromptCachePayload(
            cache_len=cache_len,
            c4_slots=c4_slots,
            c128_slots=c128_slots,
            c4_state=self.req_to_c4_state.buffer[:, req_idx].detach().clone() if self.n_c4 > 0 else None,
            c4_state_pool=self.req_to_c4_state_pool.buffer[:, req_idx].detach().clone() if self.n_c4 > 0 else None,
            c4_indexer_state=self.req_to_c4_indexer_state.buffer[:, req_idx].detach().clone()
            if self.n_c4 > 0
            else None,
            c4_indexer_state_pool=self.req_to_c4_indexer_state_pool.buffer[:, req_idx].detach().clone()
            if self.n_c4 > 0
            else None,
            swa=swa_payload,
        )

    def detach_prompt_cache_payload_from_req(self, req_idx: int, payload: DeepseekV4PromptCachePayload):
        if payload is not None and self.mem_manager is not None:
            self.mem_manager.detach_swa_for_prompt_cache(req_idx, payload.swa)
        return

    def free_prompt_cache_payload(self, payload: DeepseekV4PromptCachePayload):
        if payload is None or self.mem_manager is None:
            return
        if payload.c4_slots is not None and len(payload.c4_slots) > 0:
            self.mem_manager.free_c4(payload.c4_slots)
        if payload.c128_slots is not None and len(payload.c128_slots) > 0:
            self.mem_manager.free_c128(payload.c128_slots)
        self.mem_manager.free_swa_prompt_cache(payload.swa)
        return

    def release_prompt_cache_detached_swa(
        self,
        payload: DeepseekV4PromptCachePayload,
        keep_payload: Optional[DeepseekV4PromptCachePayload] = None,
    ):
        if payload is None or payload.swa is None or self.mem_manager is None:
            return
        old_swa = payload.swa
        if keep_payload is None or keep_payload.swa is None:
            self.mem_manager.free_swa_prompt_cache(old_swa)
            return

        old_slots = old_swa["swa_slots"].long()
        keep_slots = keep_payload.swa["swa_slots"].long()
        if old_slots.numel() == 0:
            return
        if keep_slots.numel() == 0:
            self.mem_manager.free_swa_prompt_cache(old_swa)
            return

        release_mask = ~torch.isin(old_slots, keep_slots)
        if not release_mask.any():
            return
        release_payload = {
            "full_slots": old_swa["full_slots"][release_mask].clone(),
            "swa_slots": old_swa["swa_slots"][release_mask].clone(),
        }
        self.mem_manager.free_swa_prompt_cache(release_payload)
        return

    def _reset_c128_for_prompt_cache(self, req_idx: int):
        if self.n_c128 > 0:
            self._reset_compress_cache_req(self.req_to_c128_state, req_idx)
            self._reset_state_pool_req(self.req_to_c128_state_pool, req_idx)
        return

    def rebuild_runtime_state_for_req(self, req_idx: int):
        state_map = self._runtime_states[req_idx]
        state_map.clear()
        for layer_index, ratio in enumerate(self.compress_rates):
            if ratio == 4:
                cstate_kv, cstate_score = self.get_compress_state_for_req(layer_index, req_idx)
                idx_state = self.get_c4_indexer_compress_state(layer_index)
                state_map[layer_index] = {
                    "cstate_kv": cstate_kv,
                    "cstate_score": cstate_score,
                    "idx_cstate_kv": idx_state[req_idx, 0],
                    "idx_cstate_score": idx_state[req_idx, 1],
                }
            elif ratio == 128:
                cstate_kv, cstate_score = self.get_compress_state_for_req(layer_index, req_idx)
                state_map[layer_index] = {
                    "cstate_kv": cstate_kv,
                    "cstate_score": cstate_score,
                }
        return

    def restore_prompt_cache_payload(self, req_idx: int, payload: DeepseekV4PromptCachePayload):
        assert self.mem_manager is not None
        cache_len = int(payload.cache_len)
        c4_count = cache_len // 4
        c128_count = cache_len // 128
        if c4_count > 0:
            assert payload.c4_slots is not None and len(payload.c4_slots) == c4_count
            self.req_to_c4_indexs[req_idx, :c4_count] = payload.c4_slots.cuda(non_blocking=True)
        if c128_count > 0:
            assert payload.c128_slots is not None and len(payload.c128_slots) == c128_count
            self.req_to_c128_indexs[req_idx, :c128_count] = payload.c128_slots.cuda(non_blocking=True)
        self._c4_entry_counts[req_idx] = c4_count
        self._c128_entry_counts[req_idx] = c128_count

        if self.n_c4 > 0:
            if payload.c4_state is None or payload.c4_indexer_state is None:
                raise RuntimeError("DeepSeek-V4 prompt cache hit is missing c4 running state")
            self.req_to_c4_state.buffer[:, req_idx].copy_(payload.c4_state)
            self.req_to_c4_indexer_state.buffer[:, req_idx].copy_(payload.c4_indexer_state)
            if payload.c4_state_pool is not None:
                self.req_to_c4_state_pool.buffer[:, req_idx].copy_(payload.c4_state_pool)
            if payload.c4_indexer_state_pool is not None:
                self.req_to_c4_indexer_state_pool.buffer[:, req_idx].copy_(payload.c4_indexer_state_pool)
        self._reset_c128_for_prompt_cache(req_idx)
        self.mem_manager.restore_swa_from_prompt_cache(payload.swa)
        self.rebuild_runtime_state_for_req(req_idx)
        return

    def pop_prompt_cache_free_compress_indices(
        self,
        req_idx: int,
        keep_len: int,
        duplicate_start_len: Optional[int] = None,
        duplicate_end_len: Optional[int] = None,
    ):
        def collect(table, cur_count, ratio):
            ranges = []
            if duplicate_start_len is not None and duplicate_end_len is not None:
                dup_start = duplicate_start_len // ratio
                dup_end = duplicate_end_len // ratio
                if dup_end > dup_start:
                    ranges.append((dup_start, dup_end))
            keep_count = keep_len // ratio
            if cur_count > keep_count:
                ranges.append((keep_count, cur_count))
            parts = [table[req_idx, s:e].clone() for s, e in ranges if e > s]
            return torch.cat(parts, dim=0) if parts else None

        c4 = collect(self.req_to_c4_indexs, self._c4_entry_counts[req_idx], 4)
        c128 = collect(self.req_to_c128_indexs, self._c128_entry_counts[req_idx], 128)
        if self._c4_entry_counts[req_idx] > 0:
            self.req_to_c4_indexs[req_idx, : self._c4_entry_counts[req_idx]].fill_(0)
        if self._c128_entry_counts[req_idx] > 0:
            self.req_to_c128_indexs[req_idx, : self._c128_entry_counts[req_idx]].fill_(0)
        self._c4_entry_counts[req_idx] = 0
        self._c128_entry_counts[req_idx] = 0
        return c4, c128

    def free(
        self,
        free_req_indexes,
        free_token_index,
        free_c4_index=None,
        free_c128_index=None,
    ):
        """释放 dense 槽(基类)+ 压缩槽。压缩槽由调用方(infer batch)从 req_to_c*_indexs 收集后传入,
        与基类用 free_token_index 传 dense 槽的方式一致。"""
        for req_index in free_req_indexes:
            self.clear_runtime_state(req_index)
        super().free(free_req_indexes, free_token_index)
        self.free_compress_indices(free_c4_index=free_c4_index, free_c128_index=free_c128_index)
        return

    def free_req(self, free_req_index: int):
        self.clear_runtime_state(free_req_index)
        c4, c128 = self.pop_compress_indices_for_req(free_req_index)
        self.free_compress_indices(free_c4_index=c4, free_c128_index=c128)
        return super().free_req(free_req_index)

    def free_all(self):
        super().free_all()
        self._runtime_states = [{} for _ in range(self.max_request_num + 1)]
        self._c4_entry_counts = [0 for _ in range(self.max_request_num + 1)]
        self._c128_entry_counts = [0 for _ in range(self.max_request_num + 1)]
        if self.n_c4 > 0:
            self.req_to_c4_indexs.fill_(0)
            self.req_to_c4_state.buffer.fill_(0)
            self.req_to_c4_indexer_state.buffer.fill_(0)
            self.req_to_c4_state_pool.buffer.fill_(0)
            self.req_to_c4_indexer_state_pool.buffer.fill_(0)
        if self.n_c128 > 0:
            self.req_to_c128_indexs.fill_(0)
            self.req_to_c128_state.buffer.fill_(0)
            self.req_to_c128_state_pool.buffer.fill_(0)
        self._init_all_score_state()
        return
