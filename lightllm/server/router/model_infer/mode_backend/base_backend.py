import os
import numpy as np
import torch
import time
import threading
import torch.distributed as dist
from typing import List, Tuple, Callable, Optional
from transformers.configuration_utils import PretrainedConfig
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.log_utils import init_logger
from lightllm.models import get_model
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.model_infer.infer_batch import InferReq, InferReqUpdatePack
from lightllm.server.router.token_load import TokenLoad
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, InferStateLock
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.core.objs.io_objs import AbortedReqCmd
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_dp_size
from lightllm.utils.dist_utils import get_dp_world_size, get_global_dp_rank, get_current_rank_in_dp
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_node, get_node_world_size
from lightllm.utils.dist_utils import get_dp_rank_in_node, create_new_group_for_current_node
from lightllm.distributed import dist_group_manager
from .chuncked_prefill_state import ChunkedPrefillState
from lightllm.server.router.shm_reqs_io_buffer import ShmReqsIOBuffer
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager, OverlapEventPack


class ModeBackend:
    def __init__(self) -> None:
        self.shm_req_manager = ShmReqManager()

        # 当子类处于chuncked prefill 相关模式时，会使用该管理变量进行一些 chuncked prefill
        # 的推理控制，具体使用方式可以参考 ChunkedPrefillBackend 类中的使用方式。如果是非
        # chuncked prefill 相关的模式，该状态变量不会生效。
        self.chunked_prefill_state = ChunkedPrefillState()
        self.overlap_event_manager = OverlapEventManager()

        # prefill_mask_func 和 decode_mask_func 用于控制在采样输出前，通过对logics的调整，改变输出的选择空间，
        # 主要是为约束输出模式进行定制的操作
        self.prefill_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        self.decode_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        # extra_post_req_handle_func 用于添加请求InferReq的状态变化中添加额外的后处理信息，主要是状态机相关的调整等。
        self.extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None
        pass

    def init_model(self, kvargs):
        self.args: StartArgs = kvargs.get("args", None)
        assert self.args is not None
        # p d 分离模式下会有特殊的一些初始化, 所以需要传递
        # 模式参数到模型的初始化过程中进行控制
        self.run_mode = self.args.run_mode
        self.is_multimodal = False
        self.nnodes = self.args.nnodes
        self.node_rank = self.args.node_rank
        self.world_size = kvargs["world_size"]
        self.dp_size = self.args.dp
        # dp_size_in_node 计算兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = max(1, self.dp_size // self.nnodes)
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.disable_chunked_prefill = self.args.disable_chunked_prefill
        self.chunked_prefill_size = self.args.chunked_prefill_size
        self.return_all_prompt_logprobs = self.args.return_all_prompt_logprobs
        self.use_dynamic_prompt_cache = not self.args.disable_dynamic_prompt_cache
        self.eos_id: List[int] = kvargs.get("eos_id", [2])
        self.disable_cudagraph = self.args.disable_cudagraph

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
        self.model: TpPartBaseModel = self.model  # for easy typing
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

        # 初始化 dp 模式使用的通信 tensor, 对于非dp模式，不会使用到
        if self.dp_size > 1:
            self.dp_reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_gather_item_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_all_gather_tensor = torch.tensor(
                [0 for _ in range(self.global_world_size)], dtype=torch.int32, device="cuda", requires_grad=False
            )

        self.node_broadcast_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        self.node_nccl_group = create_new_group_for_current_node("nccl")

        self.init_custom()

        self.shm_reqs_io_buffer = ShmReqsIOBuffer()

        # 启动infer_loop_thread
        self.infer_loop_thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_loop_thread.start()
        self.infer_loop_thread1 = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_loop_thread1.start()
        return

    def init_custom(self):
        pass

    def get_max_total_token_num(self):
        return self.model.mem_manager.size

    def infer_loop(self):
        raise NotImplementedError()

    def prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        raise NotImplementedError()

    def decode(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        raise NotImplementedError()

    def _try_read_new_reqs(self):
        if self.is_master_in_node:
            if self.shm_reqs_io_buffer.is_ready():
                self.node_broadcast_tensor.fill_(1)
            else:
                self.node_broadcast_tensor.fill_(0)
        dist.broadcast(self.node_broadcast_tensor, src=0, group=self.node_nccl_group, async_op=False)
        new_buffer_is_ready = self.node_broadcast_tensor.detach().item()
        if new_buffer_is_ready:
            cmds: List = self.shm_reqs_io_buffer.read_obj()
            self.shm_reqs_io_buffer.sub_state()
            if cmds:
                if isinstance(cmds[0], AbortedReqCmd):
                    for obj in cmds:
                        if obj.req_id in g_infer_context.requests_mapping:
                            req: InferReq = g_infer_context.requests_mapping[obj.req_id]
                            req.infer_aborted = True
                else:
                    self._init_reqs(reqs=cmds)
                    self.chunked_prefill_state.need_prefill_count += 1
        return

    # 一些可以复用的通用功能函数
    def _init_reqs(self, reqs: List[Tuple]):
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
        g_infer_context.add_reqs(reqs)
        g_infer_state_lock.release()
        req_ids = [e[0] for e in reqs]
        return req_ids

    # 一些可以复用的通用功能函数
    def _get_classed_reqs(
        self,
        req_ids: List[int] = None,
        no_decode: bool = False,
        strict_prefill: bool = False,
        recover_paused: bool = False,
    ):
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
        1. wait_pause_reqs 因为推理资源不够，等待被暂停的请求。
        2. paused_reqs 已经被暂停的请求，可能会被恢复。
        3. finished_reqs 需要释放的请求
        4. prefill_reqs 需要进行prefill操作的请求
        5. decode_reqs 需要进行decode操作的请求
        """

        if req_ids is None:
            req_ids = g_infer_context.infer_req_ids

        wait_pause_reqs = []
        paused_reqs = []
        finished_reqs = []
        prefill_reqs = []
        decode_reqs = []

        # 一次性最多暂停请求的数量, 防止盲目暂停大量请求
        # 因为部分请求释放占用的token容量后，就会使推理可以正常进行。
        # 如果因为一次推理容量不足，就以当前token容量的判断暂停了大量
        # 请求，其逻辑是不适合的。
        pause_max_req_num = 2
        wait_pause_count = 0

        # 因为会使用到 radix cache 和 mem_manager 的计数信息
        # 所以需要加锁保护。
        g_infer_state_lock.acquire()
        can_alloc_token_num = g_infer_context.get_can_alloc_token_num()

        for request_id in req_ids:
            req_obj: InferReq = g_infer_context.requests_mapping[request_id]

            if req_obj.filter_mark:
                finished_reqs.append(req_obj)
                continue

            if req_obj.wait_pause:
                wait_pause_reqs.append(req_obj)
                continue

            if req_obj.paused:
                paused_reqs.append(req_obj)
                continue

            if req_obj.infer_aborted or req_obj.finish_status.is_finished():
                req_obj.filter_mark = True
                continue

            if no_decode:
                is_decode = False
            else:
                is_decode = req_obj.cur_kv_len + 1 == req_obj.get_cur_total_len()
                if is_decode and strict_prefill and req_obj.cur_kv_len + 1 == req_obj.shm_req.input_len:
                    is_decode = False

            if is_decode:
                token_num = req_obj.decode_need_token_num()
                if token_num <= can_alloc_token_num:
                    decode_reqs.append(req_obj)
                    can_alloc_token_num -= token_num
                else:
                    if wait_pause_count < pause_max_req_num:
                        req_obj.wait_pause = True
                        wait_pause_count += 1
            else:
                token_num = req_obj.prefill_need_token_num(is_chuncked_prefill=not self.disable_chunked_prefill)
                if token_num <= can_alloc_token_num:
                    prefill_reqs.append(req_obj)
                    can_alloc_token_num -= token_num
                else:
                    if wait_pause_count < pause_max_req_num:
                        req_obj.wait_pause = True
                        wait_pause_count += 1

        g_infer_state_lock.release()

        g_infer_context.filter_reqs(finished_reqs=finished_reqs)
        g_infer_context.pause_reqs(wait_pause_reqs)

        if recover_paused:
            g_infer_context.recover_paused_reqs(paused_reqs=paused_reqs)

        return prefill_reqs, decode_reqs

    # 一些可以复用的通用功能函数
    def _pre_post_handle(self, run_reqs: List[InferReq], is_chuncked_mode: bool) -> List[InferReqUpdatePack]:
        update_func_objs: List[InferReqUpdatePack] = []
        # 通用状态预先填充
        is_master_in_dp = self.is_master_in_dp
        for req_obj in run_reqs:
            req_obj: InferReq = req_obj
            if is_chuncked_mode:
                new_kv_len = req_obj.get_chuncked_input_token_len()
            else:
                new_kv_len = req_obj.get_cur_total_len()
            req_obj.cur_kv_len = new_kv_len
            if is_master_in_dp:
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len

            # 对于没有到达需要输出 token 阶段的请求，直接略过, 说明还
            # 处于chuncked prefill kv 填充的阶段。
            if req_obj.cur_kv_len < req_obj.get_cur_total_len():
                pack = InferReqUpdatePack(req_obj=req_obj, output_len=0)
                update_func_objs.append(pack)
                continue

            # 将生成的下一个token的信息写入到管理对象中。
            req_obj.cur_output_len += 1
            pack = InferReqUpdatePack(req_obj=req_obj, output_len=req_obj.cur_output_len)
            update_func_objs.append(pack)
        return update_func_objs

    # 一些可以复用的通用功能函数
    def _post_handle(
        self,
        run_reqs: List[InferReq],
        next_token_ids: List[int],
        next_token_logprobs: List[float],
        run_reqs_update_packs: List[InferReqUpdatePack],
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
    ):
        """
        extra_post_req_handle_func 用于提供在一个请求确定输出的时候，给出额外的后处理操作，主要是用于
        约束输出等模式，设置自己请求内部的状态机的状态，并添加额外的停止判定条件等。
        """
        for req_obj, next_token_id, next_token_logprob, pack in zip(
            run_reqs, next_token_ids, next_token_logprobs, run_reqs_update_packs
        ):
            req_obj: InferReq = req_obj
            pack: InferReqUpdatePack = pack
            pack.handle(
                next_token_id=next_token_id,
                next_token_logprob=next_token_logprob,
                eos_ids=self.eos_id,
                extra_post_req_handle_func=extra_post_req_handle_func,
                is_master_in_dp=self.is_master_in_dp,
            )

        g_infer_context.req_manager.req_sampling_params_manager.update_reqs_token_counter(
            req_objs=run_reqs, next_token_ids=next_token_ids
        )
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

    # 对mtp 运行模式下的请求进行校验和过滤，保留校验成功的请求对象，并释放不再使用的kv 的 mem_index
    def _verify_mtp(self, run_reqs: List[InferReq], next_token_ids_cpu: np.ndarray, input_mem_indexes_cpu: np.ndarray):
        verify_ok_reqs = []
        verify_ok_req_indexes = []
        verify_ok_req_last_indexes = []
        need_free_mem_indexes = []
        grouped_reqs = self._group_mtp_run_reqs(run_reqs, next_token_ids_cpu, input_mem_indexes_cpu)
        for req_group in grouped_reqs:
            pre_req, pre_out_token_id, _, pre_index = req_group[0]
            verify_ok_reqs.append(pre_req)
            verify_ok_req_indexes.append(pre_index)
            need_verify = True
            verify_ok_count = 0
            for i in range(1, len(req_group)):
                cur_req, cur_out_token_id, cur_mem_index, cur_index = req_group[i]
                cur_req: InferReq = cur_req
                # cur_req 的输入，等于pre_req 的输出，表示校验成功
                if need_verify and cur_req.mtp_gen_token_ids[i - 1] == pre_out_token_id:
                    verify_ok_reqs.append(cur_req)
                    verify_ok_req_indexes.append(cur_index)
                    pre_req, pre_out_token_id, _, pre_index = (
                        cur_req,
                        cur_out_token_id,
                        cur_mem_index,
                        cur_index,
                    )
                    verify_ok_count += 1
                    continue

                need_verify = False
                need_free_mem_indexes.append(cur_mem_index)

            verify_ok_req_last_indexes.append(verify_ok_req_indexes[-1])

            # 清理每个请求上的 mtp_gen_token_ids, 并更新接受率信息
            pre_req.mtp_gen_token_ids = []
            if self.is_master_in_dp:
                pre_req.update_mtp_accepted_token_num(accept_token_num=verify_ok_count)

        return verify_ok_reqs, verify_ok_req_indexes, verify_ok_req_last_indexes, need_free_mem_indexes

    def _group_mtp_run_reqs(self, reqs: List[InferReq], next_token_ids_cpu: np.ndarray, input_mem_indexes: np.ndarray):
        if not reqs:
            return []

        grouped_reqs = []
        current_group = [(reqs[0], next_token_ids_cpu[0], input_mem_indexes[0], 0)]

        for i in range(1, len(reqs)):
            req = reqs[i]
            if req.req_id == current_group[-1][0].req_id:
                current_group.append((req, next_token_ids_cpu[i], input_mem_indexes[i], i))
            else:
                grouped_reqs.append(current_group)
                current_group = [(req, next_token_ids_cpu[i], input_mem_indexes[i], i)]

        grouped_reqs.append(current_group)
        return grouped_reqs

    def _gen_argmax_token_ids(self, model_output: ModelOutput):
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        draft_next_token_ids_gpu = torch.argmax(probs, dim=-1)
        return draft_next_token_ids_gpu, draft_next_token_ids_gpu.detach().cpu().numpy()

    def _update_reqs_mtp_gen_token_ids(self, reqs: List[InferReq], mtp_draft_next_token_ids: np.ndarray):
        for req, token_id in zip(reqs, mtp_draft_next_token_ids):
            req.mtp_gen_token_ids.append(token_id)
        return

    def _dp_all_gather_prefill_req_num(self, prefill_reqs: List[InferReq]) -> Tuple[np.ndarray, int]:
        """
        Gather the number of prefill requests across all DP ranks.
        """
        current_dp_prefill_num = len(prefill_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_prefill_num)
        dist.all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_prefill_req_nums = self.dp_all_gather_tensor.cpu().numpy()
        max_prefill_num = np.max(dp_prefill_req_nums)
        return dp_prefill_req_nums, max_prefill_num

    def _dp_all_reduce_decode_req_num(self, decode_reqs: List[InferReq]) -> int:
        """
        Reduce the number of decode requests across all DP ranks.
        """
        current_dp_decode_num = len(decode_reqs)
        self.dp_reduce_tensor.fill_(current_dp_decode_num)
        dist.all_reduce(self.dp_reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.dp_reduce_tensor.item()
        return max_decode_num

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

        if self.rank_in_node == 0:
            self.is_master_in_node = True
        else:
            self.is_master_in_node = False
        return
