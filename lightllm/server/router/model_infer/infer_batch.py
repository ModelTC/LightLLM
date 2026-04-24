import enum
import torch
import torch.distributed as dist
import numpy as np
import collections
import pickle

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from lightllm.common.req_manager import ReqManager, ReqManagerForMamba
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import (
    LinearAttPagedRadixCache,
    LinearAttPagedTreeNode,
)
from lightllm.utils.log_utils import init_logger
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.custom_kernel_utis import custom_cat
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.pd_io_struct import NIXLDecodeNodeInfo
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient

logger = init_logger(__name__)


@dataclass
class InferenceContext:
    req_manager: Union[ReqManager, ReqManagerForMamba] = None  # gpu 请求管理
    radix_cache: Union[LinearAttPagedRadixCache, RadixCache] = None
    shm_req_manager: ShmReqManager = None  # 共享内存请求对象管理
    requests_mapping: Dict[int, "InferReq"] = None
    infer_req_ids = None
    vocab_size = None
    cpu_embed_cache_client: Optional[CpuEmbedCacheClient] = None

    overlap_stream: torch.cuda.Stream = None  # 一些情况下推理进程进行异步折叠操作的异步流对象。
    cpu_kv_cache_stream: torch.cuda.Stream = None  # 用 cpu kv cache 操作的 stream
    is_linear_att_mixed_model: bool = False  # 标记模型是否是full att 混合 linear att 的混合模型。

    def register(
        self,
        backend,
        req_manager: Union[ReqManager, ReqManagerForMamba],
        radix_cache: Union[LinearAttPagedRadixCache, RadixCache],
        shm_req_manager: ShmReqManager,
        vocab_size: int,
    ):
        self.args = get_env_start_args()
        from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

        self.backend: ModeBackend = backend
        self.req_manager = req_manager
        self.req_sampling_manager = self.req_manager.req_sampling_params_manager
        self.radix_cache = radix_cache
        self.shm_req_manager = shm_req_manager

        self.requests_mapping = {}
        self.infer_req_ids = []

        self.vocab_size = vocab_size

        self.is_linear_att_mixed_model = isinstance(self.req_manager, ReqManagerForMamba)

        return

    def init_cpu_embed_cache_client(self):
        self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=False, init_shm_data=False)
        return

    def get_overlap_stream(self) -> torch.cuda.Stream:
        if self.overlap_stream is None:
            self.overlap_stream = torch.cuda.Stream()
        return self.overlap_stream

    def get_cpu_kv_cache_stream(self) -> torch.cuda.Stream:
        if self.cpu_kv_cache_stream is None:
            self.cpu_kv_cache_stream = torch.cuda.Stream()
        return self.cpu_kv_cache_stream

    def add_reqs(self, requests: List[Tuple[int, int, Any, int]], init_prefix_cache: bool = True) -> List["InferReq"]:
        req_objs = []
        request_ids = []
        for r in requests:
            r_id, r_index, multimodal_params, _ = r
            assert r_id not in self.requests_mapping.keys()
            r_obj = InferReq(
                req_id=r_id,
                req_idx=self.req_manager.alloc(),
                shm_index=r_index,
                multimodal_params=multimodal_params,
                vocab_size=self.vocab_size,
                init_prefix_cache=init_prefix_cache,
            )
            self.requests_mapping[r_id] = r_obj
            request_ids.append(r_id)
            req_objs.append(r_obj)

        self.infer_req_ids.extend(request_ids)

        # diverse mode 下，建立一组请求间的主从关系
        if get_env_start_args().diverse_mode:
            group_reqs: Dict[int, InferReq] = collections.defaultdict(lambda: [None, list()])
            for r_id in request_ids:
                req: InferReq = g_infer_context.requests_mapping[r_id]
                group_req_id = req.shm_req.group_req_id
                if req.req_id == group_req_id:
                    group_reqs[group_req_id][0] = req
                else:
                    group_reqs[group_req_id][1].append(req)

            for group_req_id, (master_req, slave_reqs) in group_reqs.items():
                master_req: InferReq = master_req
                master_req.slave_reqs.extend(slave_reqs)
                for slave_req in slave_reqs:
                    slave_req: InferReq = slave_req
                    slave_req.related_master_req = master_req

        return req_objs

    def free_a_req_mem(self, free_token_index: List, req: "InferReq"):
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
        else:
            if not self.is_linear_att_mixed_model:
                self._full_att_free_req(free_token_index=free_token_index, req=req)
            else:
                self._linear_att_free_req(free_token_index=free_token_index, req=req)
        req.cur_kv_len = 0
        req.shm_req.shm_cur_kv_len = req.cur_kv_len
        return

    def _full_att_free_req(self, free_token_index: List, req: "InferReq"):
        input_token_ids = req.get_input_token_ids()
        key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
        # .cpu() 是 流内阻塞操作
        value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()

        prefix_len, _ = self.radix_cache.insert(key, value)
        old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
        if req.shared_kv_node is not None:
            assert req.shared_kv_node.node_prefix_total_len <= prefix_len
            self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None
        return

    def _linear_att_free_req(self, free_token_index: List, req: "InferReq"):
        assert g_infer_context.is_linear_att_mixed_model is True
        args = get_env_start_args()
        hash_page_size = args.linear_att_hash_page_size
        big_page_num = args.linear_att_page_block_num
        shared_kv_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        tail_big_page_token_num = (
            req.linear_att_cache_len // (hash_page_size * big_page_num) * (hash_page_size * big_page_num)
        )
        page_num = req.linear_att_cache_len // hash_page_size
        assert req.linear_att_cache_len >= shared_kv_len
        if req.linear_att_cache_buffer_id is not None:
            assert req.linear_att_cache_len <= req.cur_kv_len

        if req.cur_kv_len == 0:
            return

        if req.linear_att_cache_len <= req.cur_kv_len and req.linear_att_cache_buffer_id is not None:
            # 只有小页可以有 linear_att_cache_buffer_id，然后进行小页插入。
            assert page_num % big_page_num != 0
            free_token_index.append(
                self.req_manager.req_to_token_indexs[req.req_idx][req.linear_att_cache_len : req.cur_kv_len]
            )
            req.cur_kv_len = req.linear_att_cache_len
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
            block_hashs = req.shm_req.linear_att_token_hash_list.get_all()[:page_num]
            linear_idxs = [None for _ in range(page_num)]
            linear_idxs[-1] = req.linear_att_cache_buffer_id
            req.linear_att_cache_buffer_id = None
            prefix_len, _ = self.radix_cache.insert(key, value, block_hashs=block_hashs, block_linear_idxs=linear_idxs)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None
            return

        if shared_kv_len < tail_big_page_token_num <= req.cur_kv_len:
            free_token_index.append(
                self.req_manager.req_to_token_indexs[req.req_idx][tail_big_page_token_num : req.cur_kv_len]
            )
            req.cur_kv_len = tail_big_page_token_num

            assert req.linear_att_cache_buffer_id is None
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
            cur_page_num = tail_big_page_token_num // hash_page_size
            assert tail_big_page_token_num % hash_page_size == 0
            block_hashs = req.shm_req.linear_att_token_hash_list.get_all()[:cur_page_num]
            linear_idxs = [None for _ in range(cur_page_num)]
            prefix_len, _ = self.radix_cache.insert(key, value, block_hashs=block_hashs, block_linear_idxs=linear_idxs)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None
            return

        if shared_kv_len <= req.cur_kv_len:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][shared_kv_len : req.cur_kv_len])
            req.cur_kv_len = shared_kv_len
            assert req.linear_att_cache_buffer_id is None
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.node_prefix_total_len == req.cur_kv_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None
            return

        assert False, f"error state: cur_kv_len: {req.cur_kv_len}"

    def _save_promptcache_kvbuffer(self):
        """
        save prompt cache kv buffer
        这个接口是用于保存非量化的缓存prompt cache资源，是定制场景使用的接口，当前代码中不会有调用。
        其保存的 kv 会配合量化推理模式, 加载到量化推理的prompt cache中, 提升量化推理的精度。
        like paper:
        https://arxiv.org/abs/2403.01241
        """
        prompt_cache_token_id = list(self.radix_cache.root_node.children.values())[0].token_id_key
        print(f"prompt_cache_token_id : {prompt_cache_token_id}")
        index = range(len(prompt_cache_token_id))
        prompt_cache_kv_buffer = self.radix_cache.mem_manager.get_index_kv_buffer(index)
        torch.save(prompt_cache_kv_buffer, f"prompt_cache_rank_{dist.get_rank()}.pt")

    @torch.no_grad()
    def _filter(self, finished_request_ids: List[int]):
        if len(finished_request_ids) == 0:
            return

        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req: InferReq = self.requests_mapping.pop(request_id)
            if self.args.diverse_mode:
                req.clear_master_slave_state()
            self.free_a_req_mem(free_token_index, req)

            free_req_index.append(req.req_idx)
            # logger.info(f"infer release req id {req.shm_req.request_id}")
            req.shm_req.shm_infer_released = True
            self.shm_req_manager.put_back_req_obj(req.shm_req)

        free_token_index = custom_cat(free_token_index)
        self.req_manager.free(free_req_index, free_token_index)

        finished_req_ids_set = set(finished_request_ids)
        self.infer_req_ids = [_id for _id in self.infer_req_ids if _id not in finished_req_ids_set]

        if self.radix_cache is not None and len(self.infer_req_ids) == 0:
            logger.debug(
                f"free a batch state:\n"
                f"radix refed token num {self.radix_cache.get_refed_tokens_num()}\n"
                f"radix hold token num {self.radix_cache.get_tree_total_tokens_num()}\n"
                f"mem manager can alloc token num {self.req_manager.mem_manager.can_use_mem_size}\n"
                f"mem manager total size {self.req_manager.mem_manager.size}"
            )

        return

    def filter_reqs(self, finished_reqs: List["InferReq"]):
        if finished_reqs:
            g_infer_state_lock.acquire()
            self._filter([req.req_id for req in finished_reqs])
            g_infer_state_lock.release()
        return

    @torch.no_grad()
    def pause_reqs(self, pause_reqs: List["InferReq"], is_master_in_dp: bool):
        if pause_reqs:
            g_infer_state_lock.acquire()

            free_token_index = []
            for req in pause_reqs:
                if self.args.diverse_mode:
                    # 发生暂停的时候，需要清除 diverse 模式下的主从关系
                    req.clear_master_slave_state()
                self.free_a_req_mem(free_token_index, req)
                assert req.wait_pause is True
                req.wait_pause = False
                req.paused = True
                if is_master_in_dp:
                    req.shm_req.is_paused = True
                    logger.debug(f"infer paused req id {req.req_id}")

            if len(free_token_index) != 0:
                free_token_index = custom_cat(free_token_index)
                self.req_manager.free_token(free_token_index)

            g_infer_state_lock.release()
        return self

    def recover_paused_reqs(self, paused_reqs: List["InferReq"], is_master_in_dp: bool, can_alloc_token_num: int):
        if paused_reqs:
            g_infer_state_lock.acquire()

            for req in paused_reqs:
                prefill_need_token_num = req.get_cur_total_len()
                if prefill_need_token_num > can_alloc_token_num:
                    break

                if g_infer_context.is_linear_att_mixed_model:
                    req._linear_match_radix_cache()
                else:
                    req._match_radix_cache()

                assert req.paused is True
                req.paused = False
                if is_master_in_dp:
                    req.shm_req.is_paused = False
                    logger.debug(f"infer recover paused req id {req.req_id}")
                can_alloc_token_num -= prefill_need_token_num

            g_infer_state_lock.release()
        return

    def get_can_alloc_token_num(self):
        radix_cache_unref_token_num = 0
        if self.radix_cache is not None:
            radix_cache_unref_token_num = (
                self.radix_cache.get_tree_total_tokens_num() - self.radix_cache.get_refed_tokens_num()
            )
        return self.req_manager.mem_manager.can_use_mem_size + radix_cache_unref_token_num

    def copy_linear_att_state_to_cache_buffer(
        self, b_req_idx: torch.Tensor, b_seq_len: torch.Tensor, reqs: List["InferReq"]
    ):
        """
        该函数用于在线性混合模型prefill后,如果存在大页匹配的情况下，将线性层状态复制到
        """
        if not self.is_linear_att_mixed_model:
            return
        from lightllm.common.basemodel.triton_kernel.linear_att_copy import copy_linear_att_state_to_kv_buffer

        assert len(b_req_idx) == len(b_seq_len)
        big_page_token_num = self.args.linear_att_hash_page_size * self.args.linear_att_page_block_num
        copy_linear_att_state_to_kv_buffer(
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            req_to_token_mem_index=self.req_manager.req_to_token_indexs,
            gpu_conv_state=self.req_manager.req_to_conv_state.buffer,
            gpu_ssm_state=self.req_manager.req_to_ssm_state.buffer,
            cpu_kv_conv_state=self.req_manager.mem_manager.conv_state_buffer,
            cpu_kv_ssm_state=self.req_manager.mem_manager.ssm_state_buffer,
            mtp_step=self.args.mtp_step,
            big_page_token_num=big_page_token_num,
        )

        assert not self.args.disable_chunked_prefill, "chunked prefill mode must be enabled for linear att mixed model"

        for req in reqs:
            # 判断本次prefill 完以后 kv 的长度是否到达linear att 块存储的临界点。
            if req.get_chuncked_input_token_len() == req.linear_att_cache_len:
                assert req.linear_att_cache_buffer_id is None
                if req.linear_att_cache_len % big_page_token_num != 0:
                    self.radix_cache.free_one_linear_buffer()
                    req.linear_att_cache_buffer_id = self.radix_cache.linear_att_cache_manager.alloc_one_state_cache()
                    if req.linear_att_cache_buffer_id is not None:
                        src_buffer_idx = req.req_idx * (self.args.mtp_step + 1)
                        gpu_conv_state = self.req_manager.req_to_conv_state.buffer[:, src_buffer_idx, ...]
                        gpu_ssm_state = self.req_manager.req_to_ssm_state.buffer[:, src_buffer_idx, ...]
                        dst_buffer_idx = req.linear_att_cache_buffer_id

                        dst_conv_state, dst_ssm_state = self.radix_cache.linear_att_cache_manager.get_state_cache(
                            buffer_idx=dst_buffer_idx
                        )
                        # TODO 对于非连续对象调用 copy_ 效率并不高
                        dst_conv_state.copy_(gpu_conv_state, non_blocking=True)
                        dst_ssm_state.copy_(gpu_ssm_state, non_blocking=True)
        return


g_infer_context = InferenceContext()


class InferSamplingParams:
    def __init__(
        self,
        shm_req: Req,
        vocab_size: int,
    ) -> None:
        self.shm_param = shm_req.sample_params
        self.disable_prompt_cache = self.shm_param.disable_prompt_cache
        if self.shm_param.top_k == -1:
            self.shm_param.top_k = vocab_size

        # output constraint states
        self.regular_constraint = self.shm_param.regular_constraint.to_str()
        self.guided_grammar = self.shm_param.guided_grammar.to_str()
        self.guided_json = self.shm_param.guided_json.to_str()
        if len(self.regular_constraint) == 0:
            self.regular_constraint = None
        if len(self.guided_grammar) == 0:
            self.guided_grammar = None
        if len(self.guided_json) == 0:
            self.guided_json = None

        self.fsm_current_state: int = 0
        self.allowed_token_ids = self.shm_param.allowed_token_ids.to_list()
        if len(self.allowed_token_ids) == 0:
            self.allowed_token_ids = None

        # p d mode use params
        if self.shm_param.move_kv_to_decode_node.exists:
            self.move_kv_to_decode_node = self.shm_param.move_kv_to_decode_node.to_dict()
        else:
            self.move_kv_to_decode_node = None

        # this check is not very good to placed here. to do...
        if self.allowed_token_ids is not None:
            if not all(e < vocab_size for e in self.allowed_token_ids):
                logger.error("allowed_token_ids contain tokenid >= vobsize, we remove these token ids")
                self.allowed_token_ids = [e for e in self.allowed_token_ids if e < vocab_size]

        # nixl decode node information
        if self.shm_param.nixl_params.data_len > 0:
            self.nixl_decode_node: NIXLDecodeNodeInfo = pickle.loads(self.shm_param.nixl_params.get())
        else:
            self.nixl_decode_node: NIXLDecodeNodeInfo = None

        # only pd mode used.
        self.pd_master_node_id: int = self.shm_param.pd_master_node_id.get()
        return

    def has_constraint_setting(self) -> bool:
        return (
            self.regular_constraint is not None
            or self.allowed_token_ids is not None
            or self.guided_grammar is not None
            or self.guided_json is not None
        )


class InferReq:
    class _CpuCacheTaskStatus(enum.Enum):
        NOT_STARTED = 0
        RUNNING = 1
        FINISHED = 2

        def is_not_started(self):
            return self == self.NOT_STARTED

        def is_running(self):
            return self == self.RUNNING

        def is_finished(self):
            return self == self.FINISHED

    def __init__(
        self,
        req_id: int,
        req_idx: int,
        shm_index: int,
        multimodal_params: MultimodalParams,
        vocab_size: int = -1,
        init_prefix_cache: bool = True,
    ):
        self.args = get_env_start_args()
        self.req_id = req_id
        self.req_idx = req_idx
        self.shm_index = shm_index
        self.multimodal_params = multimodal_params
        self.vocab_size = vocab_size

        # 请求需要被暂停
        self.wait_pause = False
        # 请求已经被暂停
        self.paused = False

        self.infer_aborted = False
        self.filter_mark = False
        self.need_out_token_id_statistics = True
        self.out_token_id_count: Dict[int, int] = None

        # diverse mode 下，用于标记请求组之间的依赖关系
        self.slave_reqs: List[InferReq] = []
        self.related_master_req: InferReq = None

        # nixl pd 分离模式使用的变量, 普通模式下这些变量没有具体用途
        self.nixl_trans_kv_start_index: int = 0
        self.nixl_pd_task_num: int = 0
        self.nixl_pd_task_sunccess_num: int = 0
        self.nixl_pd_task_failed_num: int = 0
        self.nixl_trans_device_id: int = -1

        # 类似 qwen3.5 这种混合linear att 模型使用的状态，记录申请来用于保存对应的线性att缓存的 buffer id
        # 当 prefill 阶段结束后, 对应长度的 linear att state 会写入到申请 buffer id 对应的块中， 方便插入到 radix cache中
        # 方便被后续的请求使用，因为这种资源是有限的，也可能不存在的情况，申请不到时, 为None，则这种小块对应长度的 kv 无法
        # 在后续被插入到radix cache中.
        self.linear_att_cache_buffer_id: Optional[int] = None
        # linear cache 对应的长度位置。
        self.linear_att_cache_len: Optional[int] = None

        # 在开启 enable_cpu_cache 的情况下，当请求结束后，会将请求的 kv cache
        # 卸载到 cpu cache 中，该标志变量用于标记请求的卸载任务的状态
        self.cpu_cache_task_status: "InferReq._CpuCacheTaskStatus" = InferReq._CpuCacheTaskStatus.NOT_STARTED

        # mtp_step 用来记录一个请求 draft模型每步需要生成的token数量
        # 正常模式下，这个值为0，在 mtp 模式下，这个值为 draft 模型每步需要生成的token数量
        self.mtp_step: int = get_env_start_args().mtp_step
        if self.mtp_step > 0:
            self.decode_need_token_num = self._mtp_decode_need_token_num
        else:
            self.decode_need_token_num = self._normal_decode_need_token_num

        if g_infer_context.is_linear_att_mixed_model:
            self.get_chuncked_input_token_len = self.get_chuncked_input_token_len_for_linear_att
            self.get_chuncked_input_token_ids = self.get_chuncked_input_token_ids_for_linear_att

        self._init_all_state()

        self.generator = None
        if self.sampling_param.shm_param.seed != -1:
            self.generator = torch.Generator(device="cuda")
            self.generator.manual_seed(self.sampling_param.shm_param.seed)

        if init_prefix_cache:
            if g_infer_context.is_linear_att_mixed_model:
                self._linear_match_radix_cache()
            else:
                self._match_radix_cache()
        return

    def _init_all_state(self):
        self.shm_req = g_infer_context.shm_req_manager.get_req_obj_by_index(self.shm_index)
        self.shm_req.link_prompt_ids_shm_array()
        self.shm_req.link_logprobs_shm_array()
        self.sampling_param: InferSamplingParams = InferSamplingParams(self.shm_req, self.vocab_size)

        # 更新 nixl pd 分离模式下， prefill 节点需要开始传输的起始位置
        if self.sampling_param.nixl_decode_node is not None:
            self.nixl_trans_kv_start_index = self.sampling_param.nixl_decode_node.ready_kv_len

        self.cur_kv_len = 0
        self.cur_output_len = 0

        g_infer_context.req_manager.req_sampling_params_manager.init_req_sampling_params(self)

        self.stop_sequences = self.sampling_param.shm_param.stop_sequences.to_list()
        # token healing mode 才被使用的管理对象
        if self.shm_req.prefix_token_ids.size != 0:
            self.prefix_token_ids = self.shm_req.prefix_token_ids.get_token_ids()
        else:
            self.prefix_token_ids = []
        self.multimodal_params = self.multimodal_params.to_dict()
        self.shared_kv_node: Union[TreeNode, LinearAttPagedTreeNode] = None

        self.finish_status = FinishStatus()

        # 申请线性att混合模型使用的缓存资源
        if g_infer_context.is_linear_att_mixed_model:
            linear_block_num = self.shm_req.linear_att_token_hash_list.size
            self.linear_att_cache_len = linear_block_num * self.args.linear_att_hash_page_size
        return

    def _match_radix_cache(self):
        assert (
            g_infer_context.is_linear_att_mixed_model is False
        ), "current _match_radix_cache does not support linear att hybrid model, to do..."
        enable_prompt_cache = (not self.sampling_param.disable_prompt_cache) and g_infer_context.radix_cache is not None
        if enable_prompt_cache and self.get_cur_total_len() > 1 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
            share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
            if share_node is not None:
                self.shared_kv_node = share_node
                ready_cache_len = share_node.node_prefix_total_len
                # 从 cpu 到 gpu 是流内阻塞操作
                g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
        return

    def _linear_match_radix_cache(self):
        assert (
            g_infer_context.is_linear_att_mixed_model is True
        ), "current _linear_match_radix_cache only support linear att hybrid model, to do..."
        enable_prompt_cache = (not self.sampling_param.disable_prompt_cache) and g_infer_context.radix_cache is not None
        linear_hash_list = self.shm_req.linear_att_token_hash_list.get_all()
        linear_att_hash_page_size = self.args.linear_att_hash_page_size
        match_tokens = min(len(linear_hash_list) * linear_att_hash_page_size, self.get_cur_total_len() - 1)
        match_tokens = max(0, match_tokens)
        match_tokens = (match_tokens // linear_att_hash_page_size) * linear_att_hash_page_size
        match_block_num = match_tokens // linear_att_hash_page_size
        linear_hash_list = linear_hash_list[:match_block_num]
        assert len(linear_hash_list) == self.shm_req.linear_att_token_hash_list.size
        big_page_token_num = linear_att_hash_page_size * self.args.linear_att_page_block_num
        big_page_is_disable = big_page_token_num > self.args.max_req_total_len
        if enable_prompt_cache and match_tokens > 1 and len(linear_hash_list) > 0 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids[0:match_tokens], dtype=torch.int64, device="cpu")
            assert len(key) == len(linear_hash_list) * linear_att_hash_page_size
            share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(
                key, block_hashs=linear_hash_list, update_refs=True
            )
            if share_node is not None:
                assert self.linear_att_cache_buffer_id is None
                if share_node.is_big_page_node():
                    # 大页匹配
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    # 从 cpu 到 gpu 是流内阻塞操作
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                    self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度
                    assert self.linear_att_cache_buffer_id is None
                    # 恢复linear att 状态
                    g_infer_context.req_manager.copy_kv_buffer_to_linear_att_state(req=self)
                else:
                    # 小页匹配
                    if big_page_is_disable:
                        # 如果 大页本质是被禁用的，可以直接使用小页的匹配结果
                        self.shared_kv_node = share_node
                        ready_cache_len = share_node.node_prefix_total_len
                        # 从 cpu 到 gpu 是流内阻塞操作
                        g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                        self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                        self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度
                        assert self.linear_att_cache_buffer_id is None
                        # 恢复linear att 状态
                        g_infer_context.req_manager.copy_cache_to_linear_att_state(
                            req=self, linear_att_cache_manager=g_infer_context.radix_cache.linear_att_cache_manager
                        )
                    else:
                        # 如果 大页本质是被启用的，则需要使用小页的匹配结果, 将小页的kv 复制到的新申请的kv位置，同时释放
                        # 对应的小页对应的节点，递归找到对应最近的大叶节点进行返回,然后赋值到req.shared_node 对象上
                        shared_kv_len = share_node.node_prefix_total_len
                        cur_big_page_tokens = (shared_kv_len // big_page_token_num) * big_page_token_num
                        need_tokens = shared_kv_len - cur_big_page_tokens
                        radix_cache = g_infer_context.radix_cache
                        if g_infer_context.get_can_alloc_token_num() > need_tokens:
                            # 有充足的token 容量时
                            radix_cache.free_radix_cache_to_get_enough_token(need_token_num=need_tokens)
                            tail_mems = radix_cache.mem_manager.alloc(need_size=need_tokens)
                            g_infer_context.req_manager.req_to_token_indexs[
                                self.req_idx, 0:cur_big_page_tokens
                            ] = value_tensor[0:cur_big_page_tokens]
                            g_infer_context.req_manager.req_to_token_indexs[
                                self.req_idx, cur_big_page_tokens:shared_kv_len
                            ] = tail_mems

                            # TODO 将 对应的 value_tensors 中的数据 拷贝到 tail_mems 中对应的数据去
                            radix_cache.mem_manager.kv_buffer[:, tail_mems, ...].copy_(
                                radix_cache.mem_manager.kv_buffer[
                                    :, value_tensor[cur_big_page_tokens:shared_kv_len], ...
                                ]
                            )

                            self.shared_kv_node = share_node  # 只是为了保证 copy_cache_to_linear_att_state 正确调用
                            g_infer_context.req_manager.copy_cache_to_linear_att_state(
                                req=self, linear_att_cache_manager=g_infer_context.radix_cache.linear_att_cache_manager
                            )
                            self.shared_kv_node = None

                            big_page_shared_node = radix_cache.deref_to_first_big_page_node(node=share_node)
                            self.shared_kv_node = big_page_shared_node
                            self.cur_kv_len = int(shared_kv_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                            self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度
                        else:
                            # 没有充足的token 容量时， 直接找到最接近的大页，进行大页恢复
                            share_node = radix_cache.deref_to_first_big_page_node(node=share_node)
                            if share_node is not None:
                                assert share_node.is_big_page_node()
                                # 大页匹配
                                self.shared_kv_node = share_node
                                ready_cache_len = share_node.node_prefix_total_len
                                # 从 cpu 到 gpu 是流内阻塞操作
                                g_infer_context.req_manager.req_to_token_indexs[
                                    self.req_idx, 0:ready_cache_len
                                ] = value_tensor[0:ready_cache_len]
                                self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64，用 int(*)转换
                                self.shm_req.prompt_cache_len = self.cur_kv_len  # 记录 prompt cache 的命中长度
                                assert self.linear_att_cache_buffer_id is None
                                # 恢复linear att 状态
                                g_infer_context.req_manager.copy_kv_buffer_to_linear_att_state(req=self)

        self.shm_req.shm_cur_kv_len = self.cur_kv_len

        if self.cur_kv_len == 0:
            # 说明没有任何命中
            g_infer_context.req_manager.init_linear_att_state(req=self)
        return

    def is_master_req(self):
        """
        diverse 模式下，判断当前请求是否为独立主请求，其进行prefill后，将
        kv 通过 radix cache 共享给其他 slave 请求， 共享后 slave 请求也
        会升级为 master 请求，具有独立推理，暂停的特性。
        """
        return self.related_master_req is None

    def is_slave_req(self):
        return self.related_master_req is not None

    def clear_master_slave_state(self):
        if self.is_slave_req():
            self.remove_master_req()
        elif self.is_master_req():
            # 数组需要 copy 后遍历。
            for slave_req in self.slave_reqs.copy():
                slave_req.remove_master_req()

    def remove_master_req(self):
        """
        一个处于 slave 状态的请求，解除与 master 请求的依赖关系后，自己会升级为
        master_req 的状态，具有独立推理，暂停的特性。
        """
        master_req = self.related_master_req
        if master_req is not None:
            master_req.slave_reqs.remove(self)
            self.related_master_req = None
        else:
            logger.warning(f"try to remove master req, but related_master_req is None, req id {self.req_id}")

    def get_radix_cache_shared_len(self):
        return 0 if self.shared_kv_node is None else self.shared_kv_node.node_prefix_total_len

    def get_output_len(self):
        return self.cur_output_len

    def get_cur_total_len(self):
        return self.shm_req.input_len + self.cur_output_len

    def get_input_token_ids(self):
        return self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]

    def get_chuncked_input_token_ids(self):
        chunked_start = self.cur_kv_len
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.args.chunked_prefill_size)
        return self.shm_req.shm_prompt_ids.arr[0:chunked_end]

    def get_chuncked_input_token_ids_for_linear_att(self):
        big_page_token_num = self.args.linear_att_hash_page_size * self.args.linear_att_page_block_num

        chunked_start = self.cur_kv_len
        chunked_end = chunked_start + self.args.chunked_prefill_size
        big_page_end = ((chunked_start // big_page_token_num) + 1) * big_page_token_num
        total_end = self.get_cur_total_len()
        end = min(total_end, chunked_end, big_page_end)

        if chunked_start < self.linear_att_cache_len < end:
            # linear att cache 对应需要存储的部分。
            end = self.linear_att_cache_len

        return self.shm_req.shm_prompt_ids.arr[0:end]

    def get_chuncked_input_token_len(self):
        chunked_start = self.cur_kv_len
        chunked_end = min(self.get_cur_total_len(), chunked_start + self.args.chunked_prefill_size)
        return chunked_end

    def get_chuncked_input_token_len_for_linear_att(self):
        big_page_token_num = self.args.linear_att_hash_page_size * self.args.linear_att_page_block_num
        chunked_start = self.cur_kv_len
        chunked_end = chunked_start + self.args.chunked_prefill_size
        big_page_end = ((chunked_start // big_page_token_num) + 1) * big_page_token_num
        total_end = self.get_cur_total_len()
        end = min(total_end, chunked_end, big_page_end)
        if chunked_start < self.linear_att_cache_len < end:
            end = self.linear_att_cache_len
        return end

    def set_next_gen_token_id(self, next_token_id: int, logprob: float, output_len: int):
        index = self.shm_req.input_len + output_len
        self.shm_req.shm_prompt_ids.arr[index - 1] = next_token_id
        self.shm_req.shm_logprobs.arr[index - 1] = logprob
        return

    def update_mtp_accepted_token_num(self, accept_token_num: int):
        # 用于统计 mtp 的接受率
        self.shm_req.mtp_accepted_token_num += accept_token_num

    def get_last_gen_token(self):
        return self.shm_req.shm_prompt_ids.arr[self.shm_req.input_len + self.cur_output_len - 1]

    def update_finish_status(self, eos_ids, output_len: int):
        if self._stop_sequences_matched(output_len=output_len):
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif (
            output_len > 0
            and self.shm_req.shm_prompt_ids.arr[self.shm_req.input_len + output_len - 1] in eos_ids
            and self.sampling_param.shm_param.ignore_eos is False
        ):
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif output_len >= self.sampling_param.shm_param.max_new_tokens:
            self.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
        return

    def _stop_sequences_matched(self, output_len: int):
        for stop_token_ids in self.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if output_len >= stop_len:
                    input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : (self.shm_req.input_len + output_len)]
                    if all(input_token_ids[i] == stop_token_ids[i] for i in range(-1, -(stop_len + 1), -1)):
                        return True
        return False

    def prefill_need_token_num(self, is_chuncked_prefill: bool):
        if is_chuncked_prefill:
            input_token_ids = self.get_chuncked_input_token_ids()
        else:
            input_token_ids = self.get_input_token_ids()

        seq_len = len(input_token_ids)
        input_token_len = seq_len - self.cur_kv_len
        return input_token_len

    def decode_need_token_num(self) -> int:
        raise NotImplementedError("error")

    def _normal_decode_need_token_num(self) -> int:
        return 1

    def _mtp_decode_need_token_num(self) -> int:
        return (1 + self.mtp_step) * 2


class InferReqUpdatePack:
    """
    用于延迟InferReq的请求更新,主要是为了方便更高效的overlap机制实现。解耦
    原有post_handle 中，部分不需要确认输出，部分需要确认输出的部分，通过该
    类绑定相关处理参数，实现解耦和延迟处理。
    """

    def __init__(self, req_obj: InferReq, output_len: int):
        self.req_obj = req_obj
        self.output_len = output_len

    def handle(
        self,
        next_token_id: int,
        next_token_logprob: float,
        eos_ids: List[int],
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]],
        is_master_in_dp: bool,
        nixl_prefill_chuncked_handle_func: Optional[Callable[[InferReq, int, float, int], None]] = None,
    ):
        # nixl_prefill_chuncked_handle_func 主要是为了处理 nixl prefill 模式下
        # 分块 prefill 后，形成对应的pd 分块传输处理。
        if nixl_prefill_chuncked_handle_func is not None:
            nixl_prefill_chuncked_handle_func(self.req_obj, next_token_id, next_token_logprob, self.output_len)

        if self.output_len <= 0:
            return

        req_obj = self.req_obj
        shm_req = req_obj.shm_req
        finish_status = req_obj.finish_status
        req_obj.set_next_gen_token_id(next_token_id, next_token_logprob, self.output_len)

        # 这里提前判定的主要作用是：
        # 在 mtp mode 下，可以存在同一个 req 对象的多次处理，
        # 在这种情况下， 如果前一步接收的mtp token 已经导致了请求
        # 达到了finished 状态，后续的请求就不再进行后续的复杂流程
        # 判断和处理，但是，因为 mtp 多请求还是导致了kv 的使用，所以
        # 还是需要更新对应的 input_tokens 和 cur_kv_len 信息，否则
        # 在 filter req 的时候，容易导致kv 管理的泄露和插入radix cache
        # 的信息不完整等问题。
        if finish_status.is_finished():
            return

        # 更新判断请求的 finished 状态
        req_obj.update_finish_status(eos_ids=eos_ids, output_len=self.output_len)

        if extra_post_req_handle_func is not None:
            extra_post_req_handle_func(req_obj, next_token_id, next_token_logprob)

        if is_master_in_dp:
            # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
            # finish_token_index finish_status candetoken_out_len 是
            # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
            shm_req.shm_cur_output_len = self.output_len

            if finish_status.is_finished():
                shm_req.finish_token_index = shm_req.input_len + self.output_len - 1
                shm_req.finish_status = req_obj.finish_status

            shm_req.candetoken_out_len = self.output_len
        return
