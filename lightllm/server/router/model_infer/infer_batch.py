import enum
import torch
import torch.distributed as dist
import numpy as np
import collections
import pickle

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from lightllm.common.req_manager import ReqManager, ReqManagerForMamba
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache
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
    req_manager: ReqManager = None  # gpu 请求管理
    radix_cache: RadixCache = None
    shm_req_manager: ShmReqManager = None  # 共享内存请求对象管理
    requests_mapping: Dict[int, "InferReq"] = None
    infer_req_ids = None
    vocab_size = None
    cpu_embed_cache_client: Optional[CpuEmbedCacheClient] = None

    overlap_stream: torch.cuda.Stream = None  # 一些情况下推理进程进行异步折叠操作的异步流对象。
    cpu_kv_cache_stream: torch.cuda.Stream = None  # 用 cpu kv cache 操作的 stream
    has_recurrent_state: bool = False  # for

    def register(
        self,
        backend,
        req_manager: ReqManager,
        radix_cache: RadixCache,
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

        self.has_recurrent_state = isinstance(self.req_manager, ReqManagerForMamba)

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

    def _alloc_and_copy_req_buffers(
        self, req_manager: ReqManagerForMamba, radix_cache: HybridRadixCache, req_objs: List["InferReq"]
    ) -> None:
        if not req_objs:
            return

        if radix_cache is not None:
            radix_cache.free_radix_cache_to_get_enough_buffer(len(req_objs))

        req_idx_gpu = torch.tensor([r.req_idx for r in req_objs], device="cuda", dtype=torch.int64)
        req_manager.alloc_buffer_for_req(req_idx_gpu)

        if radix_cache is not None:
            fork_req_ids = [
                r.req_idx for r in req_objs if r.shared_kv_node is not None and r.shared_kv_node.buffer_idx is not None
            ]
            if fork_req_ids:
                src_buf_ids = [
                    r.shared_kv_node.buffer_idx
                    for r in req_objs
                    if r.shared_kv_node is not None and r.shared_kv_node.buffer_idx is not None
                ]

                cpu_mgr = getattr(req_manager, "cpu_buffer_mem_manager", None)

                if cpu_mgr is not None:
                    # CPU-offloaded path: load from CPU SHM → GPU working buffers
                    req_tensor = torch.tensor(fork_req_ids, device="cuda", dtype=torch.int64)
                    cpu_slots = torch.tensor(src_buf_ids, dtype=torch.int64)
                    gpu_dst = req_manager.req_to_buffer_index[req_tensor, 0]
                    cpu_mgr.load_to_gpu(
                        req_manager.buffer_mem_manager.conv_state_cache.buffer,
                        req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                        cpu_slots,
                        gpu_dst,
                    )
                    cpu_mgr.sync_transfer()
                else:
                    # Original GPU-to-GPU path
                    req_tensor = torch.tensor(fork_req_ids, device="cuda", dtype=torch.int32)
                    src_tensor = torch.tensor(src_buf_ids, device="cuda", dtype=torch.int32)
                    dst_buffers = req_manager.req_to_buffer_index[req_tensor[:], 0].view(-1, 1)
                    req_manager.buffer_mem_manager.fork_state_buffers(src_tensor, dst_buffers)

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

        if isinstance(self.req_manager, ReqManagerForMamba):
            self._alloc_and_copy_req_buffers(self.req_manager, self.radix_cache, req_objs)

        return req_objs

    def free_a_req_mem(self, free_token_index: List, req: "InferReq"):
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
        elif req.sampling_param.disable_radix_cache_insert:
            # Thinking mode requests: skip radix cache insertion, free all tokens directly
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len : req.cur_kv_len])
            if req.shared_kv_node is not None:
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None
        else:
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            # .cpu() 是 流内阻塞操作
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()

            prefix_len, node = self.radix_cache.insert(key, value)

            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
            if req.shared_kv_node is not None:
                assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                req.shared_kv_node = None

    def _insert_kv_for_mamba(self, free_token_index: List, req: "InferReq"):
        """Insert KV tokens into radix tree for a mamba/hybrid request.

        Returns the tree node where KV was inserted, or None if no radix cache.
        """
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][0 : req.cur_kv_len])
            return None

        input_token_ids = req.get_input_token_ids()
        key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
        value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
        prefix_len, node = self.radix_cache.insert(key, value)
        old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
        free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
        if req.shared_kv_node is not None:
            assert req.shared_kv_node.node_prefix_total_len <= prefix_len
            self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
            req.shared_kv_node = None

        return node

    def free_a_req_mem_for_mamba(self, free_token_index: List, req: "InferReq") -> bool:
        self._insert_kv_for_mamba(free_token_index, req)
        return True  # always free request's buffer back to pool

    def _free_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """释放请求的 KV cache 和 buffer 内存"""
        if self.has_recurrent_state:
            node = self._insert_kv_for_mamba(free_token_index, req)
            req_to_buffer_index = self.req_manager.req_to_buffer_index
            cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

            # Fallback: if snapshot moments failed to attach a buffer, try now
            if node is not None and node.buffer_idx is None:
                if cpu_mgr is not None:
                    self.radix_cache.free_radix_cache_to_get_enough_buffer(1)
                    if cpu_mgr.can_use_mem_size >= 1:
                        cpu_slot = cpu_mgr.alloc(1)
                        primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
                        gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                        cpu_mgr.offload_to_cpu(
                            self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                            self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                            gpu_idx,
                            cpu_slot,
                        )
                        self.radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item())
                    free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
                else:
                    primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
                    self.radix_cache.add_buffer_idx_to_node(node, primary_buffer_idx)
            else:
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
        else:
            self.free_a_req_mem(free_token_index, req)

    def _pause_req_mem_and_buffers(self, free_token_index: List, free_buffer_index: List, req: "InferReq"):
        """Free request memory during pause, transferring the Mamba buffer to the
        tree node so that recovery can find it via match_prefix."""
        if not self.has_recurrent_state or self.radix_cache is None:
            self._free_req_mem_and_buffers(free_token_index, free_buffer_index, req)
            return

        node = self._insert_kv_for_mamba(free_token_index, req)
        req_to_buffer_index = self.req_manager.req_to_buffer_index
        cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

        if node is not None and req.cur_kv_len > 0:
            primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
            if cpu_mgr is not None:
                self.radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if cpu_mgr.can_use_mem_size >= 1:
                    cpu_slot = cpu_mgr.alloc(1)
                    gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                    cpu_mgr.offload_to_cpu(
                        self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                        self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                        gpu_idx,
                        cpu_slot,
                    )
                    self.radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item())
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())
            else:
                self.radix_cache.add_buffer_idx_to_node(node, primary_buffer_idx)
                free_buffer_index.extend(req_to_buffer_index[req.req_idx, 1:].tolist())
        else:
            free_buffer_index.extend(req_to_buffer_index[req.req_idx, :].tolist())

    def _make_buffer_snapshot_fn(self, req: "InferReq"):
        """Create a buffer_snapshot_fn callback for insert_with_buffer.

        The callback handles CPU-offload or GPU-to-GPU copy depending on
        whether a CPU buffer manager is available.
        """
        radix_cache = self.radix_cache
        req_to_buffer_index = self.req_manager.req_to_buffer_index
        cpu_mgr = getattr(self.req_manager, "cpu_buffer_mem_manager", None)

        def snapshot(node):
            if node.buffer_idx is not None:
                return
            primary_buffer_idx = req_to_buffer_index[req.req_idx, 0].item()
            if cpu_mgr is not None:
                radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if cpu_mgr.can_use_mem_size < 1:
                    return  # no buffer available
                cpu_slot = cpu_mgr.alloc(1)
                gpu_idx = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                cpu_mgr.offload_to_cpu(
                    self.req_manager.buffer_mem_manager.conv_state_cache.buffer,
                    self.req_manager.buffer_mem_manager.ssm_state_cache.buffer,
                    gpu_idx,
                    cpu_slot,
                )
                radix_cache.add_buffer_idx_to_node(node, cpu_slot[0].item())
            else:
                radix_cache.free_radix_cache_to_get_enough_buffer(1)
                if radix_cache.buffer_mem_manager.can_use_mem_size < 1:
                    return  # no buffer available
                new_buf = radix_cache.buffer_mem_manager.alloc(1)
                cur_buf = torch.tensor([primary_buffer_idx], dtype=torch.int64, device="cuda")
                new_buf_cuda = new_buf.to(device="cuda", dtype=torch.int64).contiguous()
                radix_cache.buffer_mem_manager.copy_state_buffers(cur_buf, new_buf_cuda)
                radix_cache.add_buffer_idx_to_node(node, new_buf[0].item())

        return snapshot

    def _maybe_snapshot_unbuffered_prefix(self, run_reqs: List["InferReq"]):
        """Snapshot Mamba state when prefill reaches the unbuffered prefix boundary.

        The GPU Mamba state reflects cur_kv_len tokens (recurrent — state is
        cumulative), so the tree key must cover [0:cur_kv_len] to stay
        consistent with the stored state.  Using a shorter key
        (unbuffered_prefix_pos) would associate the wrong state with a prefix,
        causing accuracy degradation for future requests that match it.
        """
        reqs = [r for r in run_reqs if r.unbuffered_prefix_pos > 0 and r.cur_kv_len >= r.unbuffered_prefix_pos]
        if not reqs:
            return
        radix_cache = self.radix_cache
        for req in reqs:
            # Use cur_kv_len as key length — Mamba state is at this position
            key = torch.tensor(req.get_input_token_ids()[: req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()
            snapshot_fn = self._make_buffer_snapshot_fn(req)
            prefix_len, new_node = radix_cache.insert(key, value)
            if new_node is not None and new_node.buffer_idx is None:
                snapshot_fn(new_node)
            # Update shared_kv_node ref counting
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            if req.shared_kv_node is not None:
                radix_cache.dec_node_ref_counter(req.shared_kv_node)
            radix_cache.add_node_ref_counter(new_node)
            req.shared_kv_node = new_node
            # Free overlapping tokens
            if old_prefix_len < prefix_len:
                self.req_manager.free_token(
                    self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
                )
            req.unbuffered_prefix_pos = 0

    def _maybe_snapshot_full_input(self, run_reqs: List["InferReq"]):
        """Snapshot Mamba state for the full input after prefill completes."""
        reqs = [r for r in run_reqs if r.cur_kv_len + 1 >= r.get_cur_total_len() and r.cur_output_len == 0]
        if not reqs:
            return
        radix_cache = self.radix_cache
        for req in reqs:
            # Skip if shared node already has a buffer and covers the full input
            if req.shared_kv_node is not None and req.shared_kv_node.buffer_idx is not None:
                if req.shared_kv_node.node_prefix_total_len >= req.cur_kv_len:
                    continue
            key = torch.tensor(req.get_input_token_ids()[: req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].cpu()
            snapshot_fn = self._make_buffer_snapshot_fn(req)
            prefix_len, new_node = radix_cache.insert(key, value)
            if new_node is not None and new_node.buffer_idx is None:
                snapshot_fn(new_node)
            old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
            if req.shared_kv_node is not None:
                radix_cache.dec_node_ref_counter(req.shared_kv_node)
            radix_cache.add_node_ref_counter(new_node)
            req.shared_kv_node = new_node
            if old_prefix_len < prefix_len:
                self.req_manager.free_token(
                    self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len]
                )

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
        free_buffer_index = []
        for request_id in finished_request_ids:
            req: InferReq = self.requests_mapping.pop(request_id)
            if self.args.diverse_mode:
                req.clear_master_slave_state()
            self._free_req_mem_and_buffers(free_token_index, free_buffer_index, req)
            free_req_index.append(req.req_idx)
            # logger.info(f"infer release req id {req.shm_req.request_id}")
            req.shm_req.shm_infer_released = True
            self.shm_req_manager.put_back_req_obj(req.shm_req)

        if len(free_token_index) != 0:
            free_token_index = custom_cat(free_token_index)
            self.req_manager.free(free_req_index, free_token_index)

        if len(free_buffer_index) != 0 and isinstance(self.req_manager, ReqManagerForMamba):
            self.req_manager.free_buffer(free_buffer_index)

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
            free_buffer_index = []
            for req in pause_reqs:
                if self.args.diverse_mode:
                    # 发生暂停的时候，需要清除 diverse 模式下的主从关系
                    req.clear_master_slave_state()
                self._free_req_mem_and_buffers(free_token_index, free_buffer_index, req)
                req.cur_kv_len = 0
                req.shm_req.shm_cur_kv_len = req.cur_kv_len
                assert req.wait_pause is True
                req.wait_pause = False
                req.paused = True
                if is_master_in_dp:
                    req.shm_req.is_paused = True
                    logger.debug(f"infer paused req id {req.req_id}")

            if len(free_token_index) != 0:
                free_token_index = custom_cat(free_token_index)
                self.req_manager.free_token(free_token_index)

            if len(free_buffer_index) != 0 and isinstance(self.req_manager, ReqManagerForMamba):
                self.req_manager.free_buffer(free_buffer_index)

            g_infer_state_lock.release()
        return self

    def recover_paused_reqs(self, paused_reqs: List["InferReq"], is_master_in_dp: bool, can_alloc_token_num: int):
        if paused_reqs:
            g_infer_state_lock.acquire()
            recovered_reqs = []
            for req in paused_reqs:
                prefill_need_token_num = req.get_cur_total_len()
                if prefill_need_token_num > can_alloc_token_num:
                    break
                req._match_radix_cache()
                assert req.paused is True
                req.paused = False
                if is_master_in_dp:
                    req.shm_req.is_paused = False
                    logger.debug(f"infer recover paused req id {req.req_id}")
                can_alloc_token_num -= prefill_need_token_num
                recovered_reqs.append(req)
            if isinstance(self.req_manager, ReqManagerForMamba):
                self._alloc_and_copy_req_buffers(self.req_manager, self.radix_cache, recovered_reqs)
            g_infer_state_lock.release()
        return

    def get_can_alloc_token_num(self):
        radix_cache_unref_token_num = 0
        if self.radix_cache is not None:
            radix_cache_unref_token_num = (
                self.radix_cache.get_tree_total_tokens_num() - self.radix_cache.get_refed_tokens_num()
            )
        return self.req_manager.mem_manager.can_use_mem_size + radix_cache_unref_token_num


g_infer_context = InferenceContext()


class InferSamplingParams:
    def __init__(
        self,
        shm_req: Req,
        vocab_size: int,
    ) -> None:
        self.shm_param = shm_req.sample_params
        self.disable_prompt_cache = self.shm_param.disable_prompt_cache
        self.disable_radix_cache_insert = self.shm_param.disable_radix_cache_insert
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

        self._init_all_state()

        self.generator = None
        if self.sampling_param.shm_param.seed != -1:
            self.generator = torch.Generator(device="cuda")
            self.generator.manual_seed(self.sampling_param.shm_param.seed)

        if init_prefix_cache:
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
        self.shared_kv_node: TreeNode = None

        self.finish_status = FinishStatus()

        # Position of unbuffered prefix node (from match_prefix walk-back).
        # 0 means no mid-prefill snapshot needed.
        self.unbuffered_prefix_pos = 0
        return

    def _match_radix_cache(self):
        if self.sampling_param.disable_prompt_cache:
            return
        if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1 and self.cur_kv_len == 0:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值

            if g_infer_context.has_recurrent_state:
                share_node, kv_len, value_tensor, unbuffered_prefix_pos = g_infer_context.radix_cache.match_prefix(
                    key, update_refs=True
                )
                if share_node is not None:
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)
                    self.shm_req.prompt_cache_len = self.cur_kv_len
                    self.unbuffered_prefix_pos = self.cur_kv_len + unbuffered_prefix_pos
                elif unbuffered_prefix_pos > 0:
                    # Bootstrap: tree had KV but no buffer anywhere
                    self.unbuffered_prefix_pos = unbuffered_prefix_pos
            else:
                share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
                if share_node is not None:
                    self.shared_kv_node = share_node
                    ready_cache_len = share_node.node_prefix_total_len
                    g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                    self.cur_kv_len = int(ready_cache_len)
                    self.shm_req.prompt_cache_len = self.cur_kv_len

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
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
        chunked_end = self.get_chuncked_input_token_len()
        return self.shm_req.shm_prompt_ids.arr[0:chunked_end]

    def get_chuncked_input_token_len(self):
        chunked_start = self.cur_kv_len
        if self.unbuffered_prefix_pos > 0 and chunked_start < self.unbuffered_prefix_pos:
            # Use unbuffered_prefix_pos as the endpoint, but still respect chunk size
            # to avoid exceeding batch_max_tokens and causing a scheduling deadlock.
            chunked_end = min(
                self.get_cur_total_len(),
                self.unbuffered_prefix_pos,
                chunked_start + self.shm_req.chunked_prefill_size,
            )
        else:
            chunked_end = min(self.get_cur_total_len(), chunked_start + self.shm_req.chunked_prefill_size)
        return chunked_end

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
