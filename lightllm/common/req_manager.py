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
    """prompt cache 载荷: 只剩 swa 按页有效性 bitmap。

    槽位与 compressor 状态都不进载荷: full_to_swa/full_to_c4/full_to_c128 以 full token 槽位
    为键(radix 持有 full 槽 ⇒ 映射行存活,free 级联回收);c4/c128 compressor 状态以 swa
    页派生寻址(随 swa 页生灭,命中零拷贝续算)。c128 partial state 不跨 radix 的 128 边界保存。

    * ``swa_page_valid``: cpu bool [cache_len // page]，插入时按当下 full_to_swa 映射写定
      (页内 128 个映射全有效才为 True)。匹配层据此把命中裁剪到"结尾页有效"的 128 边界,
      swa 压力阀回收节点页时清零。"""

    cache_len: int
    swa_page_valid: Optional[torch.Tensor] = None


class DeepseekV4PromptCacheValueOps:
    def __init__(self, req_manager: "DeepseekV4ReqManager"):
        self.req_manager = req_manager

    def slice(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        return self.req_manager.slice_prompt_cache_payload(payload, start, end)

    def concat(self, payloads: List[DeepseekV4PromptCachePayload]):
        return self.req_manager.concat_prompt_cache_payloads(payloads)

    def free(self, payload: DeepseekV4PromptCachePayload):
        # 槽位资源全部由 mem_manager.free(full_slots) 级联回收，载荷本身没有需要释放的资源。
        return

    def invalidate_swa_pages(self, payload: DeepseekV4PromptCachePayload) -> None:
        """swa 压力阀回收了该节点的 swa 页后清 bitmap: 后续命中按缩短语义裁剪,不会复活。"""
        if payload is not None and payload.swa_page_valid is not None:
            payload.swa_page_valid.fill_(False)
        return

    def valid_match_length(self, payload: Optional[DeepseekV4PromptCachePayload], natural_len: int) -> int:
        """radix 匹配裁剪: 返回 <= natural_len 的最大 128 边界 L'，使结尾页(bitmap[L'/128-1])有效。

        有效性可能非单调(owner 生前从左驱逐、后续阀从尾回收)，按候选边界回查 bitmap;
        中段 invalid 页不挡更靠后的有效命中(注意力只回看最后一个窗口)。"""
        page = self.req_manager.get_prompt_cache_page_size()
        if payload is None or payload.swa_page_valid is None:
            return 0
        n_pages = min(natural_len // page, int(payload.swa_page_valid.numel()))
        valid_idx = torch.nonzero(payload.swa_page_valid[:n_pages])
        if valid_idx.numel() == 0:
            return 0
        return (int(valid_idx[-1]) + 1) * page


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
    """DeepSeek-V4 的请求级管理。

    在基类 ReqManager 之上补 V4 专有的 per-request 结构。该对象在 mem manager profile 前创建，
    所以初始化只依赖 config 派生出的 compress_rates/head_dim/indexer_head_dim/sliding_window；
    真实 mem_manager 会在 `_init_mem_manager()` 后通过 `bind_mem_manager()` 接入。

      * 压缩槽位不在本类: ``full_to_c4/c128_indexs``(mem manager)以组末 token 的 full 槽位为键。
        本类只负责 prep 阶段的分配与 scatter(``prepare_prefill_compress_slots`` /
        ``prepare_decode_compress_slots``)——必须先于 attention metadata 构建/图捕获;
        条目内容由 layer-infer 的 compressor 前向写入。
      * compressor 在途状态不在本类: c4/c128 都在 mem manager 的 swa 页派生池,
        随页生灭,命中零拷贝续算。
      * SWA 槽位分配/出窗回收(``prepare_prefill_swa`` / ``prepare_decode_swa``): 每步 prep 阶段
        为新 token 调 mem_manager.alloc_swa，并按 per-req 水位线(``_swa_evict_marks``)惰性回收
        已出窗位置的 swa 槽。水位线首次置为该请求首个 chunk 的 ready_cache_len(radix 共享前缀
        的边界)，因此共享前缀的 swa 槽永远不会被本请求回收(归 radix 经 mem_manager.free 级联释放)。
    """

    def __init__(
        self,
        max_request_num,
        max_sequence_length,
        mem_manager: Optional[DeepseekV4MemoryManager] = None,
        compress_rates: Optional[List[int]] = None,
        head_dim: Optional[int] = None,
        indexer_head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ):
        super().__init__(max_request_num, max_sequence_length, mem_manager)

        self.mem_manager = mem_manager
        if mem_manager is not None:
            if compress_rates is None:
                compress_rates = mem_manager.compress_rates
            if head_dim is None:
                head_dim = mem_manager.head_dim
            if indexer_head_dim is None:
                indexer_head_dim = mem_manager.indexer_head_dim
            if sliding_window is None:
                sliding_window = mem_manager.sliding_window
        self.sliding_window = sliding_window
        # 出窗回收水位线: -1 表示该 req 尚未见过 prefill chunk(首个 chunk 的 ready_cache_len
        # 即共享前缀边界，作为永不下探的回收下界)。
        self._swa_evict_marks = [-1 for _ in range(max_request_num + 1)]
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

        return

    def bind_mem_manager(self, mem_manager: DeepseekV4MemoryManager):
        assert isinstance(mem_manager, DeepseekV4MemoryManager)
        assert self.compress_rates == mem_manager.compress_rates
        assert self.head_dim == mem_manager.head_dim
        assert self.indexer_head_dim == mem_manager.indexer_head_dim
        if self.sliding_window is None:
            self.sliding_window = mem_manager.sliding_window
        else:
            assert mem_manager.sliding_window is None or self.sliding_window == mem_manager.sliding_window
        self.mem_manager = mem_manager
        return

    # ------------------------------------------------------------------ swa slot prep (per step)
    def _swa_retain_len(self) -> int:
        """出窗回收的保留长度 = window + 一个 radix 页。

        多留一页使「最近一个完成的 128 边界」的结尾页恒驻留: prompt cache 只能在 floor(cur/128)
        边界入树(radix page=128)，若回收只留 window，则任何非对齐时刻该边界的结尾页都已被
        部分回收，插入门会把所有插入裁到 0(prompt cache 形同虚设)。预算即 v5 §2 的每请求
        「活跃窗口跨页 ≤2」。驻留证明要求 window >= page-1(DSV4 实际 window == page == 128)。"""
        return int(self.sliding_window) + self.get_prompt_cache_page_size()

    def prepare_prefill_swa(
        self,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        b_seq_len: torch.Tensor,
    ) -> None:
        """prefill prep: 为本 chunk 全部新 token(位置 [ready, seq))分配位置对齐的 swa 槽，
        并回收已出窗位置的槽。

        本 chunk 起点 L = ready_cache_len，首个新 token(位置 L)的窗口是 [L-W+1, L]；回收
        边界再额外保留一个 radix 页(_swa_retain_len)，即位置 < L-retain+1。先回收再分配。
        必须在 init_req_to_token_indexes 之后调用(位置对齐分配经 req_to_token 行派生/scatter)。"""
        assert self.mem_manager is not None
        if self.sliding_window is not None:
            retain = self._swa_retain_len()
            evict_slots = []
            req_list = b_req_idx.detach().cpu().tolist()
            ready_list = b_ready_cache_len.detach().cpu().tolist()
            for req_idx, ready_len in zip(req_list, ready_list):
                req_idx = int(req_idx)
                if req_idx == self.HOLD_REQUEST_ID:
                    continue
                ready_len = int(ready_len)
                mark = self._swa_evict_marks[req_idx]
                if mark < 0:
                    # 首个 chunk: [0, ready_len) 是 radix 共享前缀，其 swa 槽归 radix 所有，不可回收。
                    self._swa_evict_marks[req_idx] = ready_len
                    continue
                evict_end = ready_len - retain + 1
                if evict_end > mark:
                    evict_slots.append(self.req_to_token_indexs[req_idx, mark:evict_end])
                    self._swa_evict_marks[req_idx] = evict_end
            if evict_slots:
                self.mem_manager.evict_swa(torch.cat(evict_slots))
        self.mem_manager.alloc_swa_prefill(b_req_idx, b_ready_cache_len, b_seq_len, self.req_to_token_indexs)
        return

    def prepare_decode_swa(
        self,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        mem_indexes: torch.Tensor,
    ) -> None:
        """decode prep: 回收出窗槽并为本步新 token 分配位置对齐的 swa 槽。当前 query 位置
        seq_len-1 的窗口是 [seq_len-W, seq_len-1]；回收边界额外保留一个 radix 页
        (_swa_retain_len)，即位置 < seq_len-retain。先回收再分配。"""
        assert self.mem_manager is not None
        if self.sliding_window is not None:
            retain = self._swa_retain_len()
            evict_slots = []
            req_list = b_req_idx.detach().cpu().tolist()
            seq_list = b_seq_len.detach().cpu().tolist()
            for req_idx, seq_len in zip(req_list, seq_list):
                req_idx = int(req_idx)
                if req_idx == self.HOLD_REQUEST_ID:
                    continue
                seq_len = int(seq_len)
                mark = self._swa_evict_marks[req_idx]
                if mark < 0:
                    # 未经过 prefill prep 的保守路径: 不回收旧位置，仅推进水位线。
                    self._swa_evict_marks[req_idx] = max(0, seq_len - retain)
                    continue
                evict_end = seq_len - retain
                if evict_end > mark:
                    evict_slots.append(self.req_to_token_indexs[req_idx, mark:evict_end])
                    self._swa_evict_marks[req_idx] = evict_end
            if evict_slots:
                self.mem_manager.evict_swa(torch.cat(evict_slots))
        self.mem_manager.alloc_swa_decode(b_req_idx, b_seq_len, mem_indexes, self.req_to_token_indexs)
        return

    def init_compress_state(self, req_idx: int):
        """新请求开始时重置 runtime 水位线(对应 mamba 的 init_linear_att_state 调用点)。

        c4/c128 compressor state 都随 swa 页寻址,由内核按组覆写;请求复用时不做 per-req 清零。"""
        self.clear_runtime_state(req_idx)
        return

    # ------------------------------------------------------------------ compress slot prep (per step)
    def _compress_mapping_alloc(self, ratio: int):
        assert self.mem_manager is not None, "DeepSeek-V4 mem manager is not bound yet"
        if ratio == 4:
            return self.mem_manager.full_to_c4_indexs, self.mem_manager.alloc_c4
        if ratio == 128:
            return self.mem_manager.full_to_c128_indexs, self.mem_manager.alloc_c128
        raise AssertionError(f"invalid DeepSeek-V4 compress ratio {ratio}")

    def _scatter_compress_slots(self, ratio: int, full_slots: torch.Tensor) -> None:
        """为组末 full 槽位分配压缩槽并写入映射。已映射(>=0)的行跳过——重复 prep 幂等。"""
        if full_slots.numel() == 0:
            return
        mapping, alloc = self._compress_mapping_alloc(ratio)
        full_slots = full_slots.cuda().long().reshape(-1)
        # 去重: 同批重复键会让后写覆盖先写,先分配的压缩槽成为孤儿(allocator 泄漏)。
        need = torch.unique(full_slots[mapping[full_slots] < 0])
        if need.numel() == 0:
            return
        new_slots = alloc(need.numel()).cuda(non_blocking=True).to(torch.int32)
        mapping[need] = new_slots
        return

    def prepare_prefill_compress_slots(
        self,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        b_seq_len: torch.Tensor,
    ) -> None:
        """prefill prep: 为本 chunk 内的组末 token(位置 (g+1)*ratio-1 ∈ [ready, seq))分配压缩槽，
        scatter 进 full_to_c4/c128_indexs。必须在 init_req_to_token_indexes 之后(组末 full 槽
        从 req_to_token_indexs 取)、attention metadata 构建之前调用。"""
        if self.n_c4 == 0 and self.n_c128 == 0:
            return
        req_list = b_req_idx.detach().cpu().tolist()
        ready_list = b_ready_cache_len.detach().cpu().tolist()
        seq_list = b_seq_len.detach().cpu().tolist()
        for ratio, n_layers in ((4, self.n_c4), (128, self.n_c128)):
            if n_layers == 0:
                continue
            end_slots = []
            for req_idx, ready_len, seq_len in zip(req_list, ready_list, seq_list):
                req_idx = int(req_idx)
                if req_idx == self.HOLD_REQUEST_ID:
                    continue
                first, last = int(ready_len) // ratio, int(seq_len) // ratio
                if last > first:
                    ends = self.req_to_token_indexs[req_idx, ratio - 1 : last * ratio : ratio]
                    end_slots.append(ends[first:])
            if end_slots:
                self._scatter_compress_slots(ratio, torch.cat(end_slots))
        return

    def prepare_decode_compress_slots(
        self,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        mem_indexes: torch.Tensor,
    ) -> None:
        """decode prep: 本步 token 关闭一个组(seq_len % ratio == 0)时为其分配压缩槽并 scatter。
        组末 full 槽即本步的 mem_index(此刻 req_to_token_indexs 尚未写入本步槽位)。"""
        if self.n_c4 == 0 and self.n_c128 == 0:
            return
        req_list = b_req_idx.detach().cpu().tolist()
        seq_list = b_seq_len.detach().cpu().tolist()
        for ratio, n_layers in ((4, self.n_c4), (128, self.n_c128)):
            if n_layers == 0:
                continue
            rows = [
                i
                for i, (req_idx, seq_len) in enumerate(zip(req_list, seq_list))
                if int(req_idx) != self.HOLD_REQUEST_ID and int(seq_len) > 0 and int(seq_len) % ratio == 0
            ]
            if rows:
                self._scatter_compress_slots(ratio, mem_indexes.reshape(-1)[rows])
        return

    def alloc(self):
        req_idx = super().alloc()
        if req_idx is not None:
            self.init_compress_state(req_idx)
        return req_idx

    def clear_runtime_state(self, req_idx: int):
        # swa 槽位本身由 mem_manager.free 级联回收(随 full 槽位)，这里只复位出窗水位线。
        self._swa_evict_marks[req_idx] = -1
        return

    def get_prompt_cache_value_ops(self):
        return DeepseekV4PromptCacheValueOps(self)

    def get_prompt_cache_page_size(self):
        return 128

    def compute_swa_page_valid(self, full_slots: torch.Tensor) -> torch.Tensor:
        """按当下 full_to_swa 映射给出按页有效性: full_slots [L](L 为 page 整数倍) ->
        cpu bool [L/page]，页内全部映射有效才为 True。GPU gather + 同步,测试/校验用;
        插入热路径用 swa_page_valid_from_watermark(纯 CPU,免同步)。"""
        page = self.get_prompt_cache_page_size()
        assert full_slots.numel() % page == 0
        if full_slots.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)
        swa = self.mem_manager.full_to_swa_indexs[full_slots.cuda().long().reshape(-1)]
        return (swa.view(-1, page) >= 0).all(dim=1).cpu()

    def swa_page_valid_from_watermark(self, req_idx: int, cache_len: int) -> torch.Tensor:
        """插入时的按页有效性,纯 CPU: 请求自有 token 的 swa 映射只被出窗水位线回收
        (阀不触活跃请求,级联只在 free 时),页 p 全驻留 ⟺ 页起点 128p >= 水位线。

        与 compute_swa_page_valid 在插入时刻对自有 token 等价,但不做 GPU gather/同步——
        router 关键路径上每次插入省一次对全部在途 kernel 的等待。bitmap 中借入前缀
        ([0, ready) 的页)的行在 radix insert 切片时被丢弃(既有节点保留自己的 bitmap),
        其取值无影响。"""
        page = self.get_prompt_cache_page_size()
        mark = max(0, self._swa_evict_marks[req_idx])
        n_pages = int(cache_len) // page
        return torch.arange(n_pages, dtype=torch.long) * page >= mark

    def slice_prompt_cache_payload(self, payload: DeepseekV4PromptCachePayload, start: int, end: int):
        start = int(start)
        end = int(end)
        page = self.get_prompt_cache_page_size()
        # radix page=128 保证分裂点页对齐，bitmap 可整页切分。
        return DeepseekV4PromptCachePayload(
            cache_len=end - start,
            swa_page_valid=payload.swa_page_valid[start // page : end // page].clone()
            if payload.swa_page_valid is not None
            else None,
        )

    def concat_prompt_cache_payloads(self, payloads: List[DeepseekV4PromptCachePayload]):
        if len(payloads) == 0:
            return None
        bitmaps = [p.swa_page_valid for p in payloads]
        return DeepseekV4PromptCachePayload(
            cache_len=sum(p.cache_len for p in payloads),
            swa_page_valid=torch.cat(bitmaps, dim=0) if all(b is not None for b in bitmaps) else None,
        )

    def build_prompt_cache_payload(
        self,
        req_idx: int,
        cache_len: int,
    ) -> DeepseekV4PromptCachePayload:
        """构造插入载荷。compressor 状态不进载荷(c4 随 swa 页生灭、c128 边界自然归零),
        cache_len 不再受序列末端约束——任意 128 对齐前缀皆可插入。
        swa_page_valid 不在此填: 它必须用插入时刻的映射(infer batch 在 insert 前补)。"""
        assert self.mem_manager is not None
        return DeepseekV4PromptCachePayload(cache_len=int(cache_len))

    def free(self, free_req_indexes, free_token_index):
        """dense/swa/压缩槽全部经 mem_manager.free(free_token_index) 级联回收。"""
        for req_index in free_req_indexes:
            self.clear_runtime_state(req_index)
        super().free(free_req_indexes, free_token_index)
        return

    def free_req(self, free_req_index: int):
        self.clear_runtime_state(free_req_index)
        return super().free_req(free_req_index)

    def free_all(self):
        super().free_all()
        self._swa_evict_marks = [-1 for _ in range(self.max_request_num + 1)]
        return
