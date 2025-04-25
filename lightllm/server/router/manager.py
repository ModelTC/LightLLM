import copy
import time
import uuid
import uvloop
import asyncio
import torch
import rpyc
import pickle
import threading
import inspect

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import concurrent.futures
import zmq
import zmq.asyncio
import torch.multiprocessing as mp
import torch.distributed as dist
import multiprocessing
from typing import Dict, List, Optional
from .batch import Batch
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from lightllm.utils.log_utils import init_logger, log_time_ready
from lightllm.server.router.token_load import TokenLoad
from lightllm.server.metrics.manager import MetricClient
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.common.mem_manager import ReadOnlyStaticsMemoryManager
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


class RouterManager:
    def __init__(self, args, router_port, detokenization_port, metric_port):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.node_world_size = self.world_size // args.nnodes
        self.nnodes = args.nnodes
        self.node_rank = args.node_rank
        self.dp_size = args.dp
        # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = max(1, args.dp // self.nnodes)
        self.is_multinode_tp = args.nnodes > 1 and args.dp == 1
        self.is_multinode_and_multidp = args.nnodes > 1 and args.dp > 1
        # 判断是否是保守调度，保守调度不会发生暂停 req 的情况，但是有些场景可能影响吞吐
        self.is_safe_schedule = args.router_token_ratio == 0.0
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        self.shm_req_manager = ShmReqManager()
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.read_only_statics_mem_manager = ReadOnlyStaticsMemoryManager()
        # 初始化 radix_cache_client 用于读取 prompt cache 的管理信息
        self.radix_cache_client = None

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", self.dp_size_in_node)
        for dp_index in range(self.dp_size_in_node):
            self.shared_token_load.set_estimated_peak_token_count(0, dp_index)
            self.shared_token_load.set_frozened_token_count(0, dp_index)
            self.shared_token_load.set_current_load(0.0, dp_index)
            self.shared_token_load.set_logical_max_load(0.0, dp_index)
            self.shared_token_load.set_dynamic_max_load(0.0, dp_index)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"{args.zmq_mode}127.0.0.1:{detokenization_port}")

        if self.is_multinode_tp:
            self.mulitnode_group = dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{args.nccl_host}:{args.multinode_router_gloo_port}",
                world_size=args.nnodes,
                rank=args.node_rank,
            )

        self.is_token_healing = self.args.token_healing_mode
        self.chunked_prefill_size = args.chunked_prefill_size
        self.disable_chunked_prefill = args.disable_chunked_prefill

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        self.metric_client = MetricClient(metric_port)
        self.is_pd_run_mode = self.args.run_mode in ["prefill", "decode"]
        self.is_pd_decode_mode = self.args.run_mode == "decode"
        # p d 分离模式下，需要调度锁来同步调度端和推理端的一些数据操作
        # 主要是为了防止调度失误，造成 OOM 等错误
        self.router_lock = mp.Lock()
        g_router_lock.obj = self.router_lock

        # 调度和推理进行折叠使用的线程池
        self.overlap_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.schedule_task = None
        self.overlap_event = threading.Event()
        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpc_servers = []
        # 用于 kv move 管理进程 和 推理进程进行task信息的交互。
        self.info_queue: mp.Queue = mp.Queue()
        self.mem_queues: List[torch.multiprocessing.Queue] = [
            torch.multiprocessing.Queue() for _ in range(self.node_world_size)
        ]
        self.rpc_event = multiprocessing.Event()
        self.rpc_finished_event = multiprocessing.Event()

        assert (self.world_size % self.nnodes) == 0
        node_world_size = self.world_size // self.nnodes
        for rank_id in range(self.node_rank * node_world_size, (self.node_rank + 1) * node_world_size):
            rpc_model = await start_model_process(
                args=self.args,
                rank=rank_id,
                rank_in_node=rank_id % node_world_size,
                node_world_size=node_world_size,
                rpc_event=self.rpc_event,
                rpc_finished_event=self.rpc_finished_event,
                info_queue=self.info_queue,
                mem_queue=self.mem_queues[(rank_id % node_world_size)],
                router_lock=self.router_lock,
            )
            self.model_rpc_servers.append(rpc_model)

        self.model_rpc_client = ModelRpcClient(
            model_infer_servers=self.model_rpc_servers,
            world_size=self.world_size,
            rpc_event=self.rpc_event,
            rpc_finished_event=self.rpc_finished_event,
        )

        kvargs = {
            "args": self.args,
            "rank_id": None,  # 由后续处理填充真实数据
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "weight_dir": self.model_weightdir,
            "load_way": self.load_way,
            "max_total_token_num": self.max_total_token_num,
            "mode": self.mode,
            "max_req_num": self.args.running_max_req_size + 8,
            "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
            "nccl_host": self.args.nccl_host,
            "nccl_port": self.args.nccl_port,
            "is_first_token_constraint_mode": self.args.first_token_constraint_mode,
            "disable_chunked_prefill": self.disable_chunked_prefill,
            "chunked_prefill_size": self.chunked_prefill_size,
            "is_token_healing": self.args.token_healing_mode,
            "return_all_prompt_logprobs": self.args.return_all_prompt_logprobs,
            "use_reward_model": self.args.use_reward_model,
            "disable_dynamic_prompt_cache": self.args.disable_dynamic_prompt_cache,
            "data_type": self.args.data_type,
            "eos_id": self.eos_id,
            "diverse_mode": self.args.diverse_mode,
            "graph_max_batch_size": self.args.graph_max_batch_size,
            "graph_max_len_in_batch": self.args.graph_max_len_in_batch,
            "disable_cudagraph": self.args.disable_cudagraph,
            "mem_fraction": self.args.mem_fraction,
            "batch_max_tokens": self.args.batch_max_tokens,
            "quant_type": self.args.quant_type,
            "quant_cfg": self.args.quant_cfg,
            "pd_rpyc_ports": self.args.pd_node_infer_rpyc_ports,  # 非 pd 模式可以不设置
        }

        await self.model_rpc_client.init_model(kvargs=kvargs)

        if self.max_total_token_num is None:
            self.max_total_token_num = await self.model_rpc_client.get_max_total_token_num()
            self.args.max_total_token_num = self.max_total_token_num
        if not self.args.disable_dynamic_prompt_cache:
            self.radix_cache_client = RadixCacheReadOnlyClient(
                get_unique_server_name(),
                self.max_total_token_num,
                node_world_size=self.node_world_size,
                dp_world_size=self.world_size // self.dp_size,
            )
        self.req_queue = build_req_queue(self.args, self, self.dp_size_in_node)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")

        if self.args.run_mode == "prefill":
            # 启动 prefill kv move 管理进程
            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.prefill_node_impl import (
                start_prefill_kv_move_manager_process,
            )

            start_prefill_kv_move_manager_process(self.args, self.info_queue, self.mem_queues)

        if self.args.run_mode == "decode":
            # 启动 decode kv move 管理进程
            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.decode_node_impl import (
                start_decode_kv_move_manager_process,
            )

            start_decode_kv_move_manager_process(self.args, self.info_queue, self.mem_queues)

        return

    def add_req(self, group_req_indexes: GroupReqIndexes):
        req_group = []
        for req_index in group_req_indexes.shm_req_indexes:
            req = self.shm_req_manager.get_req_obj_by_index(req_index)
            req.multimodal_params = group_req_indexes.multimodal_params
            req.start_time = group_req_indexes.time_mark
            req_group.append(req)

            logger.info(f"router recive req id {req.request_id} cost time {time.time() - req.start_time} s")
        self.req_queue.extend(req_group)
        self.send_to_detokenization.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)

        return

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    for dp_index in range(self.dp_size_in_node):
                        token_ratio1 = self.get_used_tokens(dp_index) / self.max_total_token_num
                        token_ratio2 = (
                            self.max_total_token_num
                            - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                        ) / self.max_total_token_num
                        d_i = dp_index
                        frozen_token_num = self.shared_token_load.get_frozened_token_count(d_i)
                        estimated_peak_token_count = self.shared_token_load.get_estimated_peak_token_count(d_i)
                        logger.debug(
                            f"dp_i {d_i} current batch size: {len(self.running_batch.reqs)} \n"
                            f"dp_i {d_i} paused req num: {self.req_queue.get_paused_req_num(d_i)} \n"
                            f"dp_i {d_i} frozen token num: {frozen_token_num} \n"
                            f"dp_i {d_i} estimated_peak_token_count: {estimated_peak_token_count} \n"
                            f"dp_i {d_i} token used ratio: {token_ratio1} not contain prompt cache tree unrefed token\n"
                            f"dp_i {d_i} token used ratio: {token_ratio2} contain prompt cache tree unrefed token"
                        )
                        self.metric_client.gauge_set(
                            "lightllm_batch_pause_size", self.req_queue.get_paused_req_num(d_i)
                        )
                # pd decode mode need to update token_load more frequently
                self.req_queue.update_token_load(self.running_batch, force_update=self.is_pd_decode_mode)
                self.stats_tool.print_stats()
                self.metric_client.gauge_set("lightllm_batch_current_size", len(self.running_batch.reqs))
                self.metric_client.gauge_set("lightllm_queue_size", self.req_queue.get_wait_req_num())
                self.metric_client.gauge_set(
                    "lightllm_batch_current_max_tokens",
                    int(
                        sum(self.shared_token_load.get_dynamic_max_load(d_i) for d_i in range(self.dp_size_in_node))
                        * self.max_total_token_num
                    ),
                )
            else:
                self.req_queue.update_token_load(self.running_batch, force_update=True)
                if counter_count % 300 == 0:
                    self.metric_client.gauge_set("lightllm_batch_current_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_pause_size", 0.0)
                    self.metric_client.gauge_set("lightllm_queue_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_current_max_tokens", 0.0)
                    # 60s print once
                    if log_time_ready("frozen_info", 60):
                        for dp_i in range(self.dp_size_in_node):
                            frozen_token_num = self.shared_token_load.get_frozened_token_count(dp_i)
                            estimated_peak_token_count = self.shared_token_load.get_estimated_peak_token_count(dp_i)
                            logger.debug(f"dp_i {dp_i} frozen token num: {frozen_token_num} \n")
                            logger.debug(f"dp_i {dp_i} estimated_peak_token_count: {estimated_peak_token_count} \n")

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def get_schedule_result(self, running_batch: Batch):
        if self.schedule_task is None:

            def get_new_batch():
                limit_router_queue_length = None
                if self.is_multinode_tp:
                    # 使用 all_reduce 获取最小值
                    limit_router_queue_length = len(self.req_queue.waiting_req_list)
                    limit_router_queue_length_tensor = torch.tensor(
                        limit_router_queue_length, dtype=torch.int32, device="cpu"
                    )
                    dist.all_reduce(limit_router_queue_length_tensor, op=dist.ReduceOp.MIN, group=self.mulitnode_group)
                    limit_router_queue_length = limit_router_queue_length_tensor.item()

                self.overlap_event.wait(timeout=0.020)
                self.overlap_event.clear()
                time.sleep(0.003)
                new_batch = self.req_queue.generate_new_batch(running_batch, limit_router_queue_length)
                return new_batch

            self.schedule_task = asyncio.get_running_loop().run_in_executor(self.overlap_thread_pool, get_new_batch)
            return None
        else:
            result = await self.schedule_task
            self.schedule_task = None
            return result

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = await self.get_schedule_result(self.running_batch)
            if new_batch is not None:
                self.metric_client.histogram_observe("lightllm_batch_next_size", len(new_batch.reqs))
                for req in new_batch.reqs:
                    self.metric_client.histogram_observe(
                        "lightllm_request_queue_duration_bucket", time.time() - req.start_time
                    )
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()

                # 激进调度控制
                if not self.args.disable_aggressive_schedule:
                    self.has_wait_tokens = self.max_wait_tokens

            elif self.is_multinode_and_multidp:
                # 在多节点多 dp 的模式下，如果当前 running_batch 为None, 也需要不断的调用 decode 操作，
                # 因为其他节点上的dp可能存在运行的请求，所以本节点也需要调用decode，推理后端的backend会
                # padding 一些fake的请求来使推理过程可以正常完成。主要是给 deepseekv3 这种类型的大模型
                # 使用的，其ep并行模式下需要所有节点协同。
                await self._decode_batch(self.running_batch)

            return

        # 有运行请求，当持续decode的次数到达一个阈值，或者有上次预调度的结果存在的时。
        if self.has_wait_tokens >= self.max_wait_tokens or self.schedule_task is not None:
            new_mini_batch = await self.get_schedule_result(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:

                # 激进调度控制
                if not self.args.disable_aggressive_schedule:
                    self.has_wait_tokens = self.max_wait_tokens

                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    self.running_batch.merge(new_mini_batch)
                return

        # Check if need pause some requests for decode.
        for dp_index in range(self.dp_size_in_node):
            while not self._can_decode(self.running_batch, dp_index=dp_index):
                # pause strategy
                paused_reqs = select_paused_reqs(
                    self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num, dp_index=dp_index
                )
                await self._pause_reqs(paused_reqs)
                logger.debug(f"DP index {dp_index} pasues req num: {self.req_queue.get_paused_req_num(dp_index)}")
                self.has_wait_tokens = 0

        # Decode
        self.stats_tool.count_output_tokens(self.running_batch)
        await self._decode_batch(self.running_batch)
        self._filter_runing_batch()
        self.has_wait_tokens += 1
        return

    async def _prefill_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "prefill")
        reqs = [r.to_router_rpc_obj() for r in batch.reqs]
        self.overlap_event.set()
        await self.model_rpc_client.prefill(reqs)
        batch.filter_out_finished_req(self.shm_req_manager)
        # 发个None包触发一下detokenization
        self.send_to_detokenization.send_pyobj(None, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(f"Prefill Batch: {batch.simple_log()} \n")
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "prefill"
        )
        return

    async def _decode_batch(self, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "decode")
        self.overlap_event.set()
        await self.model_rpc_client.decode()
        # 在 self.is_multinode_and_multidp 为 True 时，传入的 batch 对象可能为 None。
        if batch is not None:
            batch.filter_out_finished_req(self.shm_req_manager)
        # 发个None包触发一下detokenization
        self.send_to_detokenization.send_pyobj(None, protocol=pickle.HIGHEST_PROTOCOL)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "decode"
        )
        return

    async def _pause_reqs(self, pasue_reqs):
        pasue_req_ids = [r.request_id for r in pasue_reqs]
        await self.model_rpc_client.pause_reqs(pasue_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _can_decode(self, batch: Batch, dp_index: int):
        if self.is_pd_run_mode or self.is_safe_schedule:
            return True
        return (
            batch.get_batch_decode_need_tokens()[dp_index] + self.get_used_tokens(dp_index) <= self.max_total_token_num
        )

    def get_used_tokens(self, dp_index):
        if not self.args.disable_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                - self.radix_cache_client.get_unrefed_tokens_num(dp_index)
            )
        else:
            return self.max_total_token_num - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)

    async def loop_for_netio_req(self):
        while True:
            recv_req: GroupReqIndexes = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                self.add_req(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        return


def start_router_process(args, router_port, detokenization_port, metric_port, pipe_writer):
    # 注册 graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            metric_port=metric_port,
        )

        asyncio.run(router.wait_to_model_ready())
    except:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        logger.error(err_str)
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"Router Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
