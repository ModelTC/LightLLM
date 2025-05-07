import os
import asyncio
import numpy as np
import rpyc
import torch
import socket
from datetime import timedelta
from typing import Dict, List, Tuple, Callable, Optional
from transformers.configuration_utils import PretrainedConfig
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.models import get_model
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.model_infer.infer_batch import InferReq, InferSamplingParams
from lightllm.server.router.token_load import TokenLoad
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, InferStateLock
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs import ShmReqManager
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_dp_size
from lightllm.utils.dist_utils import get_dp_world_size, get_global_dp_rank, get_current_rank_in_dp
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_node, get_node_world_size
from lightllm.utils.dist_utils import get_dp_rank_in_node
from lightllm.distributed import dist_group_manager
import torch.distributed as dist


class ModeBackend:
    def __init__(self) -> None:
        self.shm_req_manager = ShmReqManager()
        pass

    def init_model(self, kvargs):
        self.args = kvargs.get("args", None)
        # p d 分离模式下会有特殊的一些初始化, 所以需要传递
        # 模式参数到模型的初始化过程中进行控制
        self.run_mode = "normal" if self.args is None else self.args.run_mode
        self.is_multimodal = False
        self.nnodes = self.args.nnodes
        self.node_rank = self.args.node_rank
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.dp_size = kvargs.get("dp_size", 1)
        # dp_size_in_node 计算兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = max(1, self.dp_size // self.nnodes)
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.disable_chunked_prefill = kvargs.get("disable_chunked_prefill", False)
        self.chunked_prefill_size = kvargs.get("chunked_prefill_size", None)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.use_dynamic_prompt_cache = not kvargs.get("disable_dynamic_prompt_cache", False)
        self.eos_id: List[int] = kvargs.get("eos_id", [2])
        self.disable_cudagraph = kvargs.get("disable_cudagraph", False)

        self.cache = {}
        self.logger = init_logger(__name__)

        self.weight_dir = kvargs["weight_dir"]
        # p d 分离模式，decode节点才会使用的参数
        self.pd_rpyc_ports = kvargs.get("pd_rpyc_ports", None)
        max_total_token_num = kvargs["max_total_token_num"]

        init_distributed_env(kvargs)
        self.init_rank_infos()
        group_size = (
            2 if (self.args.enable_decode_microbatch_overlap or self.args.enable_prefill_microbatch_overlap) else 1
        )
        dist_group_manager.create_groups(group_size=group_size)  # set the default group

        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", self.dp_size_in_node)

        # 为 p d 分离模式添加的全局锁管理，用于做一些同步操作。 一定需要在
        # init_process_group 之后调用
        g_infer_state_lock.obj = (
            InferStateLock(
                name=get_unique_server_name(),
                rank_in_dp=self.rank_in_dp,
                dp_rank_in_node=self.dp_rank_in_node,
                dp_world_size=self.dp_world_size,
            )
            if self.run_mode in ["prefill", "decode"]
            else None
        )
        g_infer_state_lock.dp_world_size = self.dp_world_size
        self.infer_state_lock = g_infer_state_lock
        # 防止InferStateLock 中的全局共享信息被重复异常初始化,导致同步异常的问题。
        # 所以做一次barrier等待
        dist.barrier()

        model_cfg, _ = PretrainedConfig.get_config_dict(self.weight_dir)

        model_kvargs = {
            "weight_dir": self.weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "is_token_healing": kvargs.get("is_token_healing", False),
            "return_all_prompt_logics": self.return_all_prompt_logprobs,
            "use_dynamic_prompt_cache": self.use_dynamic_prompt_cache,
            "disable_chunked_prefill": self.disable_chunked_prefill,
            "data_type": kvargs.get("data_type", "float16"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "mem_fraction": kvargs.get("mem_fraction", 0.9),
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
            "run_mode": self.run_mode,
        }
        self.model, self.is_multimodal = get_model(model_cfg, model_kvargs)
        set_random_seed(2147483647)
        self.radix_cache = (
            RadixCache(
                get_unique_server_name(),
                self.model.mem_manager.size,
                self.rank_in_node,
                mem_manager=self.model.mem_manager,
            )
            if self.use_dynamic_prompt_cache
            else None
        )

        if "prompt_cache_kv_buffer" in model_cfg:
            assert self.use_dynamic_prompt_cache
            self.preload_prompt_cache_kv_buffer(model_cfg)

        self.logger.info(f"loaded model class {self.model.__class__}")
        g_infer_context.register(
            req_manager=self.model.req_manager,
            radix_cache=self.radix_cache,
            shm_req_manager=self.shm_req_manager,
            vocab_size=self.model.vocab_size,
        )

        self.init_custom()
        return

    def init_custom(self):
        pass

    def get_max_total_token_num(self):
        return self.model.mem_manager.size

    def prefill(self, reqs: List[Tuple]):
        """This method can be overridden in subclasses."""
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=200)
    def decode(self):
        """This method can be overridden in subclasses."""
        raise NotImplementedError()

    def pause_reqs(self, req_ids):
        if self.dp_size_in_node != 1:
            req_ids = [req_id for req_id in req_ids if req_id in g_infer_context.requests_mapping]

        g_infer_context.pause_reqs(req_ids)
        return

    # 一些可以复用的通用功能函数
    def _init_reqs(self, reqs: List[Tuple], init_req_obj=True):
        """
        init_req_obj 参数用于控制是否对请求对象的进行全量初始化，如果设置为True
        在 g_infer_context.add_reqs 函数中，会进行全量初始化，包括其 kv 信息等，
        如果设置为 False，则请求对象只是创建了基础信息，需要延迟到合适的时机调用
        请求对象的完整初始化，设计这个接口的用途是用于某些追求高性能场景的cpu gpu
        折叠，降低cpu 的overhead。
        """
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            reqs = [req for req in reqs if req[3] == dp_rank_in_node]

        g_infer_state_lock.acquire()
        g_infer_context.add_reqs(reqs, init_req_obj=init_req_obj)
        g_infer_state_lock.release()
        req_ids = [e[0] for e in reqs]
        return req_ids

    # 一些可以复用的通用功能函数
    def _get_classed_reqs(self, req_ids: List[int], no_decode: bool = False, strict_prefill: bool = False):
        """
        当将参数 no_decode 设置为True后，返回的 decode_reqs 永远为空list，主要是
        PD 分离的某些backend需要用这个参数进行控制，因为P节点永远只进行Prefill,
        避免一些特殊情况，如 radix cache 命中后，只有1token需要prefill，这个判断
        条件和decode请求的分类条件相同。所以添加一个参数进行区分。

        strict_prefill参数用于控制当 cur_kv_len + 1 == input_len 时，是否将请求
        分为 prefill,当 strict_prefill 设置为True时，表示需要将这个请求分为 prefill,
        为 False 时，将这个请求分为decode。 strict_prefill 主要是用于diverse mode
        使用时，其他模式目前不使用。

        将请求分类返回:
        1. unit reqs 还未完整初始化的请求
        2. aborted_reqs aborted 的请求
        3. ok_finished_reqs 正常推理完但是还没有释放的请求
        4. prefill_reqs 需要进行prefill操作的请求
        5. decode_reqs 需要进行decode操作的请求
        """
        uinit_reqs = []
        aborted_reqs = []
        ok_finished_reqs = []
        prefill_reqs = []
        decode_reqs = []

        for request_id in req_ids:
            req_obj: InferReq = g_infer_context.requests_mapping[request_id]

            if req_obj.is_uninitialized():
                uinit_reqs.append(req_obj)
                continue

            if req_obj.shm_req.router_aborted:
                aborted_reqs.append(req_obj)
                continue

            if req_obj.finish_status.is_finished():
                ok_finished_reqs.append(req_obj)
                continue

            if no_decode:
                prefill_reqs.append(req_obj)
                continue

            is_decode = req_obj.cur_kv_len + 1 == req_obj.get_cur_total_len()

            if not is_decode:
                prefill_reqs.append(req_obj)
            else:
                if strict_prefill:
                    if req_obj.cur_kv_len + 1 == req_obj.shm_req.input_len:
                        prefill_reqs.append(req_obj)
                    else:
                        decode_reqs.append(req_obj)
                else:
                    decode_reqs.append(req_obj)

        return uinit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs

    # 一些可以复用的通用功能函数
    def _post_handle(
        self,
        run_reqs: List[InferReq],
        next_token_ids,
        next_token_logprobs,
        is_chuncked_mode: bool,
        do_filter_finished_reqs: bool,
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
    ) -> List[int]:
        """
        extra_post_req_handle_func 用于提供在一个请求确定输出的时候，给出额外的后处理操作，主要是用于
        约束输出等模式，设置自己请求内部的状态机的状态，并添加额外的停止判定条件等。
        """
        finished_req_ids = []

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj: InferReq = req_obj
            if is_chuncked_mode:
                new_kv_len = req_obj.get_chuncked_input_token_len()
            else:
                new_kv_len = req_obj.get_cur_total_len()

            req_obj.cur_kv_len = new_kv_len
            if self.is_master_in_dp:
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len

            # 这个地方主要是为了提前判断是否存在abort的情况，如果abort了
            # 直接将请求放入finished 处理队列中。
            if req_obj.is_finished_or_aborted():
                finished_req_ids.append(req_obj.shm_req.request_id)
                continue

            # 对于没有到达需要输出 token 阶段的请求，直接略过
            if req_obj.cur_kv_len < req_obj.get_cur_total_len():
                continue

            # 将生成的下一个token的信息写入到管理对象中。
            req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
            req_obj.cur_output_len += 1

            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            if extra_post_req_handle_func is not None:
                extra_post_req_handle_func(req_obj, next_token_id, next_token_logprob)

            # 判断是否已经满足生成结束条件。
            if req_obj.is_finished_or_aborted():
                finished_req_ids.append(req_obj.shm_req.request_id)

            if self.is_master_in_dp:
                # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
                # finish_token_index finish_status candetoken_out_len 是
                # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
                req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                if req_obj.finish_status.is_finished():
                    req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                    req_obj.shm_req.finish_status = req_obj.finish_status

                req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

        if do_filter_finished_reqs:
            g_infer_context.filter(finished_req_ids)
        return finished_req_ids

    # 一些可以复用的通用功能函数
    def _overlap_req_init_and_filter(
        self, uninit_reqs: List[InferReq], ok_finished_reqs: List[InferReq], clear_list=False
    ):
        if uninit_reqs or ok_finished_reqs:
            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                if ok_finished_reqs:
                    g_infer_state_lock.acquire()
                    g_infer_context.filter_reqs(ok_finished_reqs)
                    g_infer_state_lock.release()

                if uninit_reqs:
                    g_infer_state_lock.acquire()
                    self._post_init_reqs(uninit_reqs)
                    g_infer_state_lock.release()

            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

            if clear_list:
                uninit_reqs.clear()
                ok_finished_reqs.clear()

        return

    # 一些可以复用的通用功能函数
    def _post_init_reqs(self, uninit_reqs: List[InferReq]):
        """
        如req对象在调用 _init_reqs 函数时， init_req_obj 为 False，则在适当的时机调用
        _post_init_reqs 重新完成req对象的完整初始化
        """
        for req in uninit_reqs:
            req.init_all()
        return

    # 一些可以复用的通用功能函数
    def _filter_reqs(self, reqs: List[InferReq]):
        if reqs:
            g_infer_state_lock.acquire()
            g_infer_context.filter_reqs(reqs)
            g_infer_state_lock.release()
        return

    # 一些可以复用的通用功能函数
    def _trans_req_ids_to_req_objs(self, req_ids: List[int]) -> List[InferReq]:
        return [g_infer_context.requests_mapping[req_id] for req_id in req_ids]

    def preload_prompt_cache_kv_buffer(self, model_cfg):
        self.logger.info("Preload prompt cache kv buffer.")
        cur_rank = dist.get_rank()
        prompt_cache_kv_buffer_path = os.path.join(
            self.weight_dir, model_cfg["prompt_cache_kv_buffer"][f"rank_{cur_rank}"]
        )
        prompt_cache_kv_buffer = torch.load(prompt_cache_kv_buffer_path, weights_only=True, map_location="cpu")
        intact_kv_len = len(model_cfg["prompt_cache_token_ids"])
        intact_kv_index = self.radix_cache.mem_manager.alloc(intact_kv_len)
        self.radix_cache.mem_manager.load_index_kv_buffer(intact_kv_index, prompt_cache_kv_buffer)
        self.radix_cache.insert(
            torch.tensor(model_cfg["prompt_cache_token_ids"], dtype=torch.int64, device="cpu"),
            intact_kv_index,
        )
        self.radix_cache.match_prefix(
            torch.tensor(model_cfg["prompt_cache_token_ids"], dtype=torch.int64, device="cpu"), update_refs=True
        )

    def init_rank_infos(self):
        self.node_world_size = get_node_world_size()
        self.rank_in_node = get_current_rank_in_node()
        self.current_device_id = get_current_device_id()
        self.rank_in_dp = get_current_rank_in_dp()
        self.global_dp_rank = get_global_dp_rank()
        self.dp_rank_in_node = get_dp_rank_in_node()
        self.dp_world_size = get_dp_world_size()
        self.global_rank = get_global_rank()
        self.global_world_size = get_global_world_size()
        self.dp_size = get_dp_size()

        if self.nnodes > 1 and self.dp_size == 1:
            if self.rank_in_node == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False
        else:
            if self.rank_in_dp == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False
        return
