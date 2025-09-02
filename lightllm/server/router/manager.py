import time
import uvloop
import asyncio
import torch
import pickle
import inspect
import setproctitle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
import torch.multiprocessing as mp
import torch.distributed as dist
import multiprocessing
from typing import Dict, List, Optional
from .batch import Batch, Req
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from lightllm.server.core.objs.io_objs import GroupReqIndexes, AbortedReqCmd, StopStrMatchedReqCmd
from lightllm.server.core.objs import ShmReqManager, StartArgs
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from .shm_reqs_io_buffer import ShmReqsIOBuffer
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
    def __init__(self, args: StartArgs, router_port, detokenization_port, metric_port):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.node_world_size = self.world_size // args.nnodes
        self.nnodes = args.nnodes
        self.node_rank = args.node_rank
        self.dp_size = args.dp
        self.schedule_time_interval = args.schedule_time_interval  # 默认30ms 的调度周期
        # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = max(1, args.dp // self.nnodes)
        self.is_multinode_tp = args.nnodes > 1 and args.dp == 1
        self.is_multinode_tp_master = self.is_multinode_tp and args.node_rank == 0
        self.is_multinode_tp_slave = self.is_multinode_tp and args.node_rank != 0
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

        self.running_batch: Batch = None
        context = zmq.Context(2)
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

        self.metric_client = MetricClient(metric_port)
        self.is_pd_run_mode = self.args.run_mode in ["prefill", "decode"]
        self.is_pd_decode_mode = self.args.run_mode == "decode"
        # p d 分离模式下，需要调度锁来同步调度端和推理端的一些数据操作
        # 主要是为了防止调度失误，造成 OOM 等错误
        self.router_lock = mp.Lock()
        g_router_lock.obj = self.router_lock

        self.shm_reqs_io_buffer = ShmReqsIOBuffer()
        return

    async def wait_to_model_ready(self):
        # 调度使用的对象
        self.schedule_new_batch: Batch = None

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
            "disable_chunked_prefill": self.args.disable_chunked_prefill,
            "chunked_prefill_size": self.args.chunked_prefill_size,
            "is_token_healing": self.args.token_healing_mode,
            "return_all_prompt_logprobs": self.args.return_all_prompt_logprobs,
            "use_reward_model": self.args.use_reward_model,
            "disable_dynamic_prompt_cache": self.args.disable_dynamic_prompt_cache,
            "data_type": self.args.data_type,
            "eos_id": self.args.eos_id,
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

    def _get_schedule_time_interval(self):
        # dp 模式，为了更好的配平，需要更长的调度间隔，以便于能收到更多的请求
        return self.schedule_time_interval

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 100 == 0:
                    for dp_index in range(self.dp_size_in_node):
                        token_ratio1 = self.get_used_tokens(dp_index) / self.max_total_token_num
                        token_ratio2 = (
                            self.max_total_token_num
                            - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                        ) / self.max_total_token_num
                        d_i = dp_index
                        frozen_token_num = self.shared_token_load.get_frozened_token_count(d_i)
                        estimated_peak_token_count = self.shared_token_load.get_estimated_peak_token_count(d_i)
                        paused_req_num = self._get_paused_req_num_in_dp_index(dp_index=d_i)
                        logger.debug(
                            f"dp_i {d_i} current batch size: {len(self.running_batch.reqs)} \n"
                            f"dp_i {d_i} paused req num: {paused_req_num} \n"
                            f"dp_i {d_i} frozen token num: {frozen_token_num} \n"
                            f"dp_i {d_i} estimated_peak_token_count: {estimated_peak_token_count} \n"
                            f"dp_i {d_i} token used ratio: {token_ratio1} not contain prompt cache tree unrefed token\n"
                            f"dp_i {d_i} token used ratio: {token_ratio2} contain prompt cache tree unrefed token"
                        )
                        self.metric_client.gauge_set("lightllm_batch_pause_size", paused_req_num)
                # pd decode mode need to update token_load more frequently
                self.req_queue.update_token_load(self.running_batch, force_update=self.is_pd_decode_mode)
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

            await asyncio.sleep(self._get_schedule_time_interval())

    async def _step(self):
        """
        事件处理循环
        """
        # 接受新请求，并尝试调度
        await self._recv_new_reqs_and_schedule()
        # 判断是否有新请求加入推理
        # 激进调度满足，有新的推理batch就需要进行加入。
        # 或者延迟step的步数满足了当前条件，也需要进行新的推理batch的加入。
        if (self.schedule_new_batch is not None) and self.shm_reqs_io_buffer.is_empty():
            new_batch = self.schedule_new_batch
            self.schedule_new_batch = None
            self._add_new_batch_to_running_batch(new_batch=new_batch)
            await self._add_batch(new_batch)

        self._filter_reqs_from_running_batch()
        aborted_reqs = self._get_aborted_reqs_from_running_batch()
        if aborted_reqs:
            await self._aborted_reqs(aborted_reqs=aborted_reqs)
        stop_str_matched_reqs = self._get_stop_str_reqs_from_running_batch()
        if stop_str_matched_reqs:
            await self._stop_str_matched_reqs(stop_str_matched_reqs=stop_str_matched_reqs)
        return

    async def _add_batch(self, batch: Batch):
        # 添加新请求
        reqs = [r.to_router_rpc_obj() for r in batch.reqs]
        while not self.shm_reqs_io_buffer.is_empty():
            await asyncio.sleep(0.02)

        self.shm_reqs_io_buffer.write_obj(reqs)
        self.shm_reqs_io_buffer.set_ready()
        logger.debug(f"Prefill Batch: {batch.simple_log()} \n")
        return

    async def _aborted_reqs(self, aborted_reqs: List[Req]):
        cmds = [AbortedReqCmd(req_id=r.request_id) for r in aborted_reqs]
        while not self.shm_reqs_io_buffer.is_empty():
            await asyncio.sleep(0.02)

        self.shm_reqs_io_buffer.write_obj(cmds)
        self.shm_reqs_io_buffer.set_ready()
        return

    async def _stop_str_matched_reqs(self, stop_str_matched_reqs: List[Req]):
        cmds = [StopStrMatchedReqCmd(req_id=r.request_id) for r in stop_str_matched_reqs]
        while not self.shm_reqs_io_buffer.is_empty():
            await asyncio.sleep(0.02)

        self.shm_reqs_io_buffer.write_obj(cmds)
        self.shm_reqs_io_buffer.set_ready()
        return

    def _add_new_batch_to_running_batch(self, new_batch: Batch):
        if self.running_batch is None:
            self.running_batch = new_batch
        else:
            self.running_batch.merge(new_batch)
        return

    def _filter_reqs_from_running_batch(self):
        if self.running_batch is not None:
            self.running_batch.filter_out_finished_req(self.shm_req_manager)
            if self.running_batch.is_clear():
                self.running_batch = None
        return

    def _get_aborted_reqs_from_running_batch(self) -> List[Req]:
        ans = []
        if self.running_batch is None:
            return ans
        for req in self.running_batch.reqs:
            if req.is_aborted and req._router_aborted is False:
                req._router_aborted = True
                ans.append(req)
        return ans

    def _get_stop_str_reqs_from_running_batch(self) -> List[Req]:
        # to do, 多节点tp模式，暂时不能支持 stop str 匹配退出
        if self.is_multinode_tp:
            return []

        ans = []
        if self.running_batch is None:
            return ans
        for req in self.running_batch.reqs:
            if req.stop_str_matched and req._router_stop_str_matched is False:
                req._router_stop_str_matched = True
                ans.append(req)
        return ans

    def _get_paused_req_num(self) -> int:
        if self.running_batch is None:
            return 0
        else:
            count = 0
            for req in self.running_batch.reqs:
                if req.is_paused:
                    count += 1
            return count

    def _get_paused_req_num_in_dp_index(self, dp_index: int) -> int:
        if self.running_batch is None:
            return 0
        else:
            count = 0
            for req in self.running_batch.reqs:
                if req.is_paused and req.sample_params.suggested_dp_index == dp_index:
                    count += 1
            return count

    def get_used_tokens(self, dp_index):
        if not self.args.disable_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)
                - self.radix_cache_client.get_unrefed_tokens_num(dp_index)
            )
        else:
            return self.max_total_token_num - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)

    def _add_req(self, group_req_indexes: GroupReqIndexes):
        req_group = []
        for req_index in group_req_indexes.shm_req_indexes:
            req = self.shm_req_manager.get_req_obj_by_index(req_index)
            req.multimodal_params = group_req_indexes.multimodal_params
            req.start_time = group_req_indexes.time_mark
            # 附加一个私有标记变量，标记请求是否已经被router发送过abort命令给推理进程，
            # 防止反复发送abort命令给推理进程
            req._router_aborted = False
            # 作用同 _router_aborted 类似
            req._router_stop_str_matched = False
            req_group.append(req)

            logger.info(f"router recive req id {req.request_id} cost time {time.time() - req.start_time} s")
        self.req_queue.extend(req_group)
        self.send_to_detokenization.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def _generate_new_batch(self):
        # 调度的时候需要考虑当前运行的batch，和调度了但是暂时还没有推理的部分请求。
        new_batch = self.req_queue.generate_new_batch(
            Batch.merge_two_batch(self.running_batch, self.schedule_new_batch)
        )
        self.schedule_new_batch = Batch.merge_two_batch(self.schedule_new_batch, new_batch)
        return

    def _multinode_tp_generate_new_batch(self):
        try:
            dist.barrier(group=self.mulitnode_group)

            # 调度的时候需要考虑当前运行的batch，和调度了但是暂时还没有推理的部分请求。
            if self.is_multinode_tp_master:
                new_batch = self.req_queue.generate_new_batch(
                    Batch.merge_two_batch(self.running_batch, self.schedule_new_batch)
                )
                if new_batch is not None:
                    req_ids = [req.request_id for req in new_batch.reqs]
                else:
                    req_ids = []
                dist.broadcast_object_list([len(req_ids)], src=0, group=self.mulitnode_group)
                if len(req_ids) == 0:
                    new_batch = None
                else:
                    dist.broadcast_object_list(req_ids, src=0, group=self.mulitnode_group)
                    req_id_select_mark = [1 for _ in range(len(req_ids))]
                    req_id_select_mark = torch.tensor(req_id_select_mark, dtype=torch.int32, device="cpu")
                    dist.all_reduce(req_id_select_mark, op=dist.ReduceOp.MIN, group=self.mulitnode_group)
                    back_req_list = []
                    for req_id, select in zip(req_ids, req_id_select_mark.numpy()):
                        if select == 0:
                            req = new_batch.pop_req(req_id)
                            back_req_list.append(req)
                    self.req_queue.waiting_req_list = back_req_list + self.req_queue.waiting_req_list
                    if new_batch.is_clear():
                        new_batch = None
            else:
                req_nums = [None]
                dist.broadcast_object_list(req_nums, src=0, group=self.mulitnode_group)
                req_num = req_nums[0]
                if req_num == 0:
                    new_batch = None
                else:
                    req_ids = [None for _ in range(req_num)]
                    dist.broadcast_object_list(req_ids, src=0, group=self.mulitnode_group)
                    all_req_id_set = set([req.request_id for req in self.req_queue.waiting_req_list])
                    req_id_select_mark = []
                    for req_id in req_ids:
                        req_id_select_mark.append(1 if req_id in all_req_id_set else 0)
                    req_id_select_mark = torch.tensor(req_id_select_mark, dtype=torch.int32, device="cpu")
                    dist.all_reduce(req_id_select_mark, op=dist.ReduceOp.MIN, group=self.mulitnode_group)
                    select_req_ids = []
                    for req_id, select in zip(req_ids, req_id_select_mark.numpy()):
                        if select == 1:
                            select_req_ids.append(req_id)

                    select_reqs = []
                    for req_id in select_req_ids:
                        for req in self.req_queue.waiting_req_list:
                            if req.request_id == req_id:
                                select_reqs.append(req)

                    for req in select_reqs:
                        self.req_queue.waiting_req_list.remove(req)
                    if select_reqs:
                        new_batch = Batch(-1, reqs=select_reqs, dp_size_in_node=self.dp_size_in_node)
                    else:
                        new_batch = None

            self.schedule_new_batch = Batch.merge_two_batch(self.schedule_new_batch, new_batch)

            dist.barrier(group=self.mulitnode_group)
        except Exception as e:
            logger.exception(str(e))
            raise e
        return

    async def _recv_new_reqs_and_schedule(self):
        if not hasattr(self, "recv_max_count"):
            self.recv_max_count = 64

        try:
            # 一次最多从 zmq 中取 recv_max_count 个请求，防止 zmq 队列中请求数量过多导致阻塞了主循环。
            for _ in range(self.recv_max_count):
                recv_req: GroupReqIndexes = self.recv_from_httpserver.recv_pyobj(zmq.NOBLOCK)
                if isinstance(recv_req, GroupReqIndexes):
                    self._add_req(recv_req)
                else:
                    assert False, f"Error Req Inf {recv_req}"

            # 当队列中存在较多的请求时，将一次接受的数量上调
            self.recv_max_count = min(int(self.recv_max_count * 1.3), 256)

        except zmq.ZMQError:
            # 当队列已经开始清空的时候，将一次接受的数量下调
            self.recv_max_count = 64

        if self.is_multinode_tp:
            self._multinode_tp_generate_new_batch()
        else:
            if self._get_paused_req_num() == 0:
                self._generate_new_batch()
        return

    def clean_up(self):
        return


def start_router_process(args, router_port, detokenization_port, metric_port, pipe_writer):
    # 注册 graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::router_server")
    start_parent_check_thread()

    def handle_exception(loop, context):
        logger.exception(f"Router Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)

    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            metric_port=metric_port,
        )

        loop.run_until_complete(router.wait_to_model_ready())
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
    loop.run_until_complete(router.loop_for_fwd())
    return
