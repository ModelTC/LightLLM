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
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.mtp_verify import mtp_verify
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.core.objs.io_objs import AbortedReqCmd, StopStrMatchedReqCmd
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_dp_size
from lightllm.utils.dist_utils import get_dp_world_size, get_global_dp_rank, get_current_rank_in_dp
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_node, get_node_world_size
from lightllm.utils.dist_utils import get_dp_rank_in_node, create_new_group_for_current_node
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import dist_group_manager
from lightllm.server.core.objs.shm_objs_io_buffer import ShmObjsIOBuffer
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager, OverlapEventPack
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.server.pd_io_struct import NIXLChunckedTransTaskRet


class ModeBackend:
    def __init__(self) -> None:
        self.shm_req_manager = ShmReqManager()

        self.overlap_event_manager = OverlapEventManager()
        # 标识是否支持 overlap 功能，很多子类模式如 xgrammar 和 outlines 当前不支持 overlap 高性能模式
        self.support_overlap = True

        # prefill_mask_func 和 decode_mask_func 用于控制在采样输出前，通过对logics的调整，改变输出的选择空间，
        # 主要是为约束输出模式进行定制的操作
        self.prefill_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        self.decode_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        # extra_post_req_handle_func 用于添加请求InferReq的状态变化中添加额外的后处理信息，主要是状态机相关的调整等。
        self.extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None

        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap

        # 控制 _get_classed_reqs 分类的参数变量，不同的 backend 具有可能需要不同的分类运行条件。
        self.classed_req_no_decode = False
        self.classed_req_strict_prefill = True

        # nixl pd mode callback func
        self.nixl_prefill_chuncked_handle_func: Optional[Callable[[InferReq, int, float, int], None]] = None
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
        self.batch_max_tokens = self.args.batch_max_tokens
        self.eos_id: List[int] = kvargs.get("eos_id", [2])
        self.disable_cudagraph = self.args.disable_cudagraph
        self.is_multinode_tp = self.args.nnodes > 1 and self.args.dp == 1
        self.is_nixl_pd_mode = self.run_mode in ["nixl_prefill", "nixl_decode"]
        self.is_nixl_decode_mode = self.run_mode == "nixl_decode"

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

        # 用于协同读取 ShmObjsIOBuffer 中的请求信息的通信tensor和通信组对象。
        self.node_broadcast_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        self.node_nccl_group = create_new_group_for_current_node("nccl")

        # 用于在多节点tp模式下协同读取 ShmObjsIOBuffer 中的请求信息的通信tensor和通信组对象。
        if self.is_multinode_tp:
            self.multinode_tp_gather_item_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.multinode_tp_all_gather_tensor = torch.tensor(
                [0 for _ in range(self.global_world_size)], dtype=torch.int32, device="cuda", requires_grad=False
            )
            self.multinode_tp_nccl_group = dist.new_group(
                [rank for rank in range(self.global_world_size)], backend="nccl"
            )

        self.init_custom()
        self.shm_reqs_io_buffer = ShmObjsIOBuffer()
        # 只会在 nixl pd 模式下才会使用，用于上传分块传输任务是否成功。
        self.shm_nixl_trans_io_buffer = ShmObjsIOBuffer(tail_str="nixl")

        # 开启 mtp 模式，需要完成mtp model的初始化
        if self.args.mtp_mode:
            self.init_mtp_draft_model(kvargs)

        # 启动infer_loop_thread, 启动两个线程进行推理，对于具备双batch推理折叠得场景
        # 可以降低 cpu overhead，大幅提升gpu得使用率。
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

    def init_mtp_draft_model(self, main_kvargs: dict):
        # 当前只支持 deepseekv3 模式的 mtp
        self.mtp_step = self.args.mtp_step
        self.draft_models: List[Deepseek3MTPModel] = []

        os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
        for i in range(self.mtp_step):
            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            mtp_model_kvargs = {
                "weight_dir": self.args.mtp_draft_model_dir,
                "max_total_token_num": self.model.mem_manager.size,
                "load_way": main_kvargs["load_way"],
                "mode": main_kvargs["mode"],
                "max_req_num": main_kvargs.get("max_req_num", 1000),
                "max_seq_length": main_kvargs.get("max_seq_length", 1024 * 5),
                "is_token_healing": False,
                "return_all_prompt_logics": False,
                "disable_chunked_prefill": self.disable_chunked_prefill,
                "data_type": main_kvargs.get("data_type", "float16"),
                "graph_max_batch_size": main_kvargs.get("graph_max_batch_size", 16),
                "graph_max_len_in_batch": main_kvargs.get("graph_max_len_in_batch", 8196),
                "disable_cudagraph": main_kvargs.get("disable_cudagraph", False),
                "mem_fraction": main_kvargs["mem_fraction"],
                "batch_max_tokens": main_kvargs.get("batch_max_tokens", None),
                "quant_type": main_kvargs.get("quant_type", None),
                "quant_cfg": main_kvargs.get("quant_cfg", None),
                "run_mode": "normal",
                "main_model": self.model,
                "mem_layer_start": self.model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
            }

            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            assert mtp_model_cfg["model_type"] == "deepseek_v3"
            assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
            self.draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))

            self.logger.info(f"loaded mtp model class {self.draft_models[i].__class__}")
        return

    def _async_copy_next_token_infos_to_pin_mem(self, next_token_ids: torch.Tensor, next_token_logprobs: torch.Tensor):
        """
        这个函数会把next token id和logprobs保存到pinned memory中
        这样可以保障post_handle 函数可以读取到正常的输出结果。
        """
        next_token_ids_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="next_token_ids",
            gpu_tensor=next_token_ids,
        )
        next_token_logprobs_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="next_token_logprobs",
            gpu_tensor=next_token_logprobs,
        )
        return next_token_ids_cpu, next_token_logprobs_cpu

    def _try_read_new_reqs(self):
        if self.is_multinode_tp:
            self._try_read_new_reqs_multinode_tp()
        else:
            self._try_read_new_reqs_normal()
        return

    def _try_read_new_reqs_normal(self):
        if self.is_master_in_node:
            if self.shm_reqs_io_buffer.is_ready():
                self.node_broadcast_tensor.fill_(1)
            else:
                self.node_broadcast_tensor.fill_(0)

        src_rank_id = self.args.node_rank * self.node_world_size
        dist.broadcast(self.node_broadcast_tensor, src=src_rank_id, group=self.node_nccl_group, async_op=False)
        new_buffer_is_ready = self.node_broadcast_tensor.detach().item()
        if new_buffer_is_ready:
            self._read_reqs_buffer_and_init_reqs()

        # nixl pd mode 从 shm_nixl_trans_io_buffer 读取分块传输的完成进度。
        if self.is_nixl_pd_mode:
            if self.is_master_in_node:
                if self.shm_nixl_trans_io_buffer.is_ready():
                    self.node_broadcast_tensor.fill_(1)
                else:
                    self.node_broadcast_tensor.fill_(0)

            src_rank_id = self.args.node_rank * self.node_world_size
            dist.broadcast(self.node_broadcast_tensor, src=src_rank_id, group=self.node_nccl_group, async_op=False)
            new_buffer_is_ready = self.node_broadcast_tensor.detach().item()
            if new_buffer_is_ready:
                self._read_nixl_trans_io_buffer_and_update_req_status()
        return

    def _try_read_new_reqs_multinode_tp(self):
        """
        多节点tp模式下,需要协调所有rank的行为同步。
        """
        if self.shm_reqs_io_buffer.is_ready():
            self.multinode_tp_gather_item_tensor.fill_(1)
        else:
            self.multinode_tp_gather_item_tensor.fill_(0)
        dist.all_gather_into_tensor(
            self.multinode_tp_all_gather_tensor,
            self.multinode_tp_gather_item_tensor,
            group=self.multinode_tp_nccl_group,
            async_op=False,
        )
        new_buffer_is_readys = self.multinode_tp_all_gather_tensor.detach().cpu().numpy()
        new_buffer_is_ready = np.all(new_buffer_is_readys == 1)

        if new_buffer_is_ready:
            self._read_reqs_buffer_and_init_reqs()

        assert self.is_nixl_pd_mode is False
        return

    def _read_reqs_buffer_and_init_reqs(self):
        cmds: List = self.shm_reqs_io_buffer.read_obj()
        self.shm_reqs_io_buffer.sub_state()
        if cmds:
            init_reqs = []
            for obj in cmds:
                if isinstance(obj, tuple):
                    init_reqs.append(obj)
                elif isinstance(obj, (AbortedReqCmd, StopStrMatchedReqCmd)):
                    if obj.req_id in g_infer_context.requests_mapping:
                        req: InferReq = g_infer_context.requests_mapping[obj.req_id]
                        req.infer_aborted = True
                else:
                    assert False, f"error type {type(obj)}"
            if init_reqs:
                self._init_reqs(reqs=init_reqs)
        return

    def _read_nixl_trans_io_buffer_and_update_req_status(self):
        cmds: List[NIXLChunckedTransTaskRet] = self.shm_nixl_trans_io_buffer.read_obj()
        self.shm_nixl_trans_io_buffer.sub_state()
        if cmds:
            for obj in cmds:
                if obj.request_id in g_infer_context.requests_mapping:
                    req: InferReq = g_infer_context.requests_mapping[obj.request_id]
                    if obj.has_error:
                        req.nixl_pd_task_failed_num += 1
                    else:
                        req.nixl_pd_task_sunccess_num += 1
                        # nixl decode 节点需要预填充 prefill 节点发送过来的产生的首token信息，以使
                        # 推理过程可以继续。
                        if self.is_nixl_decode_mode:
                            if obj.first_gen_token_id is not None:
                                assert req.cur_output_len == 0
                                req.cur_output_len += 1
                                req_to_next_token_ids = (
                                    self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids
                                )
                                # to do 这个地方是否需要加流同步
                                req_to_next_token_ids[req.req_idx, 0:1].fill_(obj.first_gen_token_id)
                                torch.cuda.current_stream().synchronize()
                                InferReqUpdatePack(req_obj=req, output_len=req.cur_output_len).handle(
                                    next_token_id=obj.first_gen_token_id,
                                    next_token_logprob=obj.first_gen_token_logprob,
                                    eos_ids=self.eos_id,
                                    extra_post_req_handle_func=None,
                                    is_master_in_dp=self.is_master_in_dp,
                                    nixl_prefill_chuncked_handle_func=None,
                                )
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

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        """
        将错误请求从 req_ids 中过滤出来, 然后让 _get_classed_reqs 进行处理。 该函数
        主要用于在 nixl pd 分离模式下, 由子类继承重载, prefill 和 decode 节点过滤 kv 传输错误，或者 kv
        传输没有完成的请求。
        """
        return [g_infer_context.requests_mapping[request_id] for request_id in req_ids]

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
        3. finished_reqs 需要释放的请求, 包含正常结束和aborted退出的请求。
        4. prefill_reqs 需要进行prefill操作的请求
        5. decode_reqs 需要进行decode操作的请求
        """

        if req_ids is None:
            req_ids = g_infer_context.infer_req_ids

        if len(req_ids) == 0:
            return [], []

        ready_reqs = self._filter_not_ready_reqs(req_ids)
        support_overlap = self.support_overlap

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
        prefill_tokens = 0

        # 因为会使用到 radix cache 和 mem_manager 的计数信息
        # 所以需要加锁保护。
        g_infer_state_lock.acquire()
        can_alloc_token_num = g_infer_context.get_can_alloc_token_num()

        for req_obj in ready_reqs:

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
                if support_overlap:
                    # 延迟处理
                    req_obj.filter_mark = True
                    continue
                else:
                    finished_reqs.append(req_obj)
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
                if prefill_tokens + token_num > self.batch_max_tokens:
                    continue
                if token_num <= can_alloc_token_num:
                    prefill_tokens += token_num
                    prefill_reqs.append(req_obj)
                    can_alloc_token_num -= token_num
                else:
                    if wait_pause_count < pause_max_req_num:
                        req_obj.wait_pause = True
                        wait_pause_count += 1

        g_infer_state_lock.release()

        self._pre_handle_finished_reqs(finished_reqs=finished_reqs)
        g_infer_context.filter_reqs(finished_reqs=finished_reqs)

        g_infer_context.pause_reqs(wait_pause_reqs, is_master_in_dp=self.is_master_in_dp)

        if recover_paused:
            g_infer_context.recover_paused_reqs(
                paused_reqs=paused_reqs, is_master_in_dp=self.is_master_in_dp, can_alloc_token_num=can_alloc_token_num
            )

        return prefill_reqs, decode_reqs

    def _pre_handle_finished_reqs(self, finished_reqs: List[InferReq]):
        """
        给 PD 分离模式下，prefill node 使用的继承钩子函数，用于发起 kv 传输任务。
        """
        pass

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
        nixl_prefill_chuncked_handle_func: Optional[Callable[[InferReq, int, float, int], None]] = None,
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
                nixl_prefill_chuncked_handle_func=nixl_prefill_chuncked_handle_func,
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

    def _verify_mtp_v2(
        self, new_next_token_ids: torch.Tensor, b_req_idx: torch.Tensor, b_req_mtp_start_loc: torch.Tensor
    ):
        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=new_next_token_ids,
            b_req_idx=b_req_idx,
        )
        return mtp_accept_len, accepted_index

    def _update_mtp_accept_ratio(
        self,
        decode_reqs: List[InferReq],
        mtp_accept_len_cpu: torch.Tensor,
    ):
        if self.is_master_in_dp:
            for req, accept_len in zip(decode_reqs, mtp_accept_len_cpu):
                req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
        return

    def _gen_argmax_token_ids(self, model_output: ModelOutput):
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        draft_next_token_ids_gpu = torch.argmax(probs, dim=-1)
        return draft_next_token_ids_gpu

    def _dp_all_gather_prefill_and_decode_req_num(
        self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather the number of prefill requests across all DP ranks.
        """
        current_dp_prefill_num = len(prefill_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_prefill_num)
        dist.all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_prefill_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        current_dp_decode_num = len(decode_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_decode_num)
        dist.all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_decode_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        return dp_prefill_req_nums, dp_decode_req_nums

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
