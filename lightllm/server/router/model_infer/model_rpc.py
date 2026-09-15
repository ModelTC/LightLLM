import os
import asyncio
import rpyc
import socket
import uuid
import torch.multiprocessing as mp
import multiprocessing
import threading
import inspect
import setproctitle
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from rpyc.utils.factory import unix_connect
from lightllm.server.router.model_infer.mode_backend import (
    ChunkedPrefillBackend,
    FirstTokenConstraintBackend,
    OutlinesConstraintBackend,
    RewardModelBackend,
    XgrammarBackend,
    DPChunkedPrefillBackend,
    DiversehBackend,
    PDChunkedPrefillForPrefillNode,
    PDDPChunkedForPrefillNode,
    PDDecodeNode,
    PDDPForDecodeNode,
)
from lightllm.server.router.model_infer.mode_backend.redundancy_expert_manager import RedundancyExpertManager
from lightllm.server.router.model_infer.mode_backend.rl_backend_ops import RlBackendOps
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.torch_memory_saver_utils import MemoryTag
from lightllm.server.io_struct import RlOpReq, RlOpRsp

logger = init_logger(__name__)


class ModelRpcServer(rpyc.Service):
    def __init__(self, args, rank: int, rank_in_node: int, node_world_size: int, info_queue: mp.Queue):
        super().__init__()
        self.args: StartArgs = args
        self.node_world_size = node_world_size
        self.info_queue = info_queue

        self.rank = rank
        self.rank_in_node = rank_in_node
        self.backend = None
        self.rl_backend_ops = None
        logger.info(f"Initialized RPC server for rank {self.rank}.")
        return

    def exposed_init_model(self, kvargs):
        # 填充真正的 rank_id 参数
        kvargs = obtain(kvargs)
        kvargs["rank_id"] = self.rank
        self.world_size = kvargs["world_size"]
        use_reward_model = self.args.use_reward_model
        diverse_mode = self.args.diverse_mode
        is_first_token_constraint_mode = self.args.first_token_constraint_mode

        is_outlines_constraint_mode = self.args.output_constraint_mode == "outlines"
        is_xgrammar_constraint_mode = self.args.output_constraint_mode == "xgrammar"
        assert not (is_outlines_constraint_mode and is_xgrammar_constraint_mode), "only one constraint mode can be true"
        is_prefill_node = self.args.run_mode == "prefill"
        is_decode_node = self.args.run_mode == "decode"

        if is_prefill_node:
            if self.args.dp > 1:
                self.backend = PDDPChunkedForPrefillNode(self.info_queue)
            else:
                self.backend = PDChunkedPrefillForPrefillNode(self.info_queue)

        elif is_decode_node:
            if self.args.dp > 1:
                self.backend = PDDPForDecodeNode(self.info_queue)
            else:
                self.backend = PDDecodeNode(self.info_queue)

        elif self.args.dp > 1:
            self.backend = DPChunkedPrefillBackend()
        elif use_reward_model:
            self.backend = RewardModelBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
        elif is_outlines_constraint_mode:
            self.backend = OutlinesConstraintBackend()
        elif is_xgrammar_constraint_mode:
            self.backend = XgrammarBackend()
        elif is_first_token_constraint_mode:
            self.backend = FirstTokenConstraintBackend()
        else:
            self.backend = ChunkedPrefillBackend()

        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)
        self.rl_backend_ops = RlBackendOps(self.backend) if self.args.enable_rl else None

        # only deepseekv3 can support auto_update_redundancy_expert
        if self.args.auto_update_redundancy_expert:
            self.redundancy_expert_manager = RedundancyExpertManager(self.backend.model)
            logger.info("init redundancy_expert_manager")
        else:
            self.redundancy_expert_manager = None
        return

    def exposed_get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()

    def _calibration_control(self, req: RlOpReq) -> RlOpRsp:
        if not self.args.export_fp8kv_calibration:
            raise ValueError("calibration control is unavailable outside internal export mode")
        if not isinstance(req.op_args, dict) or req.op_args.get("job_id") != self.args.calibration_job_id:
            raise ValueError("calibration job_id does not match this service")
        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        infer_pending = len(g_infer_context.infer_req_ids)
        mapped_pending = len(g_infer_context.requests_mapping)
        io_pending = int(self.backend.shm_reqs_io_buffer.is_ready())
        mem = self.backend.model.mem_manager
        if req.op_name == "calibration_status":
            return RlOpRsp(
                success=True,
                msg="ok",
                op_name=req.op_name,
                op_result={"idle": not (infer_pending or mapped_pending or io_pending), **mem.calibration_status()},
            )
        if infer_pending or mapped_pending or io_pending:
            raise ValueError("calibration operation requires an idle rank")
        op = mem.begin_calibration if req.op_name == "calibration_begin" else mem.snapshot_calibration
        return RlOpRsp(success=True, msg="ok", op_name=req.op_name, op_result={"idle": True, **op()})

    def exposed_rl_op(self, req: RlOpReq) -> RlOpRsp:
        try:
            req = obtain(req)
            if req.op_name in {"calibration_status", "calibration_begin", "calibration_snapshot"}:
                return self._calibration_control(req)
            if self.rl_backend_ops is None:
                raise ValueError("RL backend ops is not initialized")
            if not RlBackendOps.supports(req.op_name):
                raise ValueError(f"Unsupported RL op {req.op_name}. Supported ops: {sorted(RlBackendOps.SUPPORTED)}")
            success, ret = self.rl_backend_ops.dispatch(req.op_name, req.op_args)
            return RlOpRsp(success=success, msg=str(ret), op_name=req.op_name, op_result=ret)
        except BaseException as e:
            logger.exception(f"rl op failed: {str(e)}")
            return RlOpRsp(success=False, msg=f"rl op failed: {str(e)}", op_name=req.op_name)


class ModelRpcClient:
    def __init__(self, conn):
        self.conn = conn

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                try:
                    ans = f(*args, **kwargs)
                except BaseException as e:
                    logger.exception(str(e))
                    os._exit(-1)

                await asyncio.to_thread(ans.wait)
                # raise if exception
                return ans.value

            return _func

        self._init_model = async_wrap(self.conn.root.init_model)
        self._get_max_total_token_num = async_wrap(self.conn.root.get_max_total_token_num)
        self._rl_op = async_wrap(self.conn.root.rl_op)
        return

    async def init_model(self, kvargs):
        ans = self._init_model(kvargs)
        await ans
        return

    async def get_max_total_token_num(self):
        ans = self._get_max_total_token_num()
        return obtain(await ans)

    async def rl_op(self, req: RlOpReq) -> RlOpRsp:
        ans = self._rl_op(req)
        return obtain(await ans)


def _init_env(
    args,
    rank,
    rank_in_node,
    node_world_size,
    info_queue,
    socket_path,
    success_event,
):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::model_infer:RANK{rank}")
    start_parent_check_thread()

    model_rpc_server = ModelRpcServer(args, rank, rank_in_node, node_world_size, info_queue)
    # Start rpyc server with Unix socket
    t = ThreadedServer(model_rpc_server, socket_path=socket_path, protocol_config={"allow_pickle": True})

    success_event.set()
    t.start()
    return


async def start_model_process(
    args,
    rank,
    rank_in_node,
    node_world_size,
    info_queue: mp.Queue,
):
    import lightllm.utils.rpyc_fix_utils as _

    socket_path = _generate_unix_socket_path()
    if os.path.exists(socket_path):
        os.remove(socket_path)

    success_event = mp.Event()
    proc = mp.Process(
        target=_init_env,
        args=(
            args,
            rank,
            rank_in_node,
            node_world_size,
            info_queue,
            socket_path,
            success_event,
        ),
    )
    # 若开启 --enable_torch_memory_saver：必须在 configure_subprocess() 内
    # 调用 proc.start()，以便子进程继承/完成 torch_memory_saver 的初始化钩子；
    # 后续 Infer 才能对 KV / weight / cudagraph 等显存做 pause/resume。
    # 未开启时 Wrapper 为空实现，with 块无额外开销。
    from lightllm.utils.torch_memory_saver_utils import TorchMemorySaverWrapper

    torch_memory_saver = TorchMemorySaverWrapper(args.enable_torch_memory_saver)
    with torch_memory_saver.configure_subprocess():
        proc.start()

    # Use asyncio.to_thread to make the blocking wait non-blocking
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    from lightllm.utils.retry_utils import retry

    conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})

    return ModelRpcClient(conn=conn)


def _generate_unix_socket_path() -> str:
    """Generate a random Unix socket path"""
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_model_infer_{unique_id}.sock"
