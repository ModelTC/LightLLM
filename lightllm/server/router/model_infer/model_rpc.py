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
    ReturnPromptLogProbBackend,
    RewardModelBackend,
    TokenHealingBackend,
    XgrammarBackend,
    DPChunkedPrefillBackend,
    DiversehBackend,
    DecodeNode,
    DPForDecodeNode,
    ChunckedPrefillForPrefillNode,
    DPChunkedForPrefillNode,
    NIXLChunckedPrefillForPrefillNode,
    NIXLDPChunkedForPrefillNode,
    NIXLDecodeNode,
    NIXLDPForDecodeNode,
)
from lightllm.server.router.model_infer.mode_backend.redundancy_expert_manager import RedundancyExpertManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.torch_memory_saver_utils import MemoryTag
from lightllm.server.io_struct import GeneralHttpToModelRpcReq, GeneralModelToHttpRpcRsp

logger = init_logger(__name__)


class ModelRpcServer(rpyc.Service):
    def __init__(self, args, rank: int, rank_in_node: int, node_world_size: int, info_queue: mp.Queue):
        super().__init__()
        self.args: StartArgs = args
        self.node_world_size = node_world_size
        self.info_queue = info_queue

        self.rank = rank
        self.rank_in_node = rank_in_node
        logger.info(f"Initialized RPC server for rank {self.rank}.")
        return

    def exposed_init_model(self, kvargs):
        # 填充真正的 rank_id 参数
        kvargs = obtain(kvargs)
        kvargs["rank_id"] = self.rank
        self.world_size = kvargs["world_size"]
        return_all_prompt_logprobs = self.args.return_all_prompt_logprobs
        use_reward_model = self.args.use_reward_model
        diverse_mode = self.args.diverse_mode
        is_token_healing = self.args.token_healing_mode
        is_first_token_constraint_mode = self.args.first_token_constraint_mode

        is_outlines_constraint_mode = self.args.output_constraint_mode == "outlines"
        is_xgrammar_constraint_mode = self.args.output_constraint_mode == "xgrammar"
        assert not (is_outlines_constraint_mode and is_xgrammar_constraint_mode), "only one constraint mode can be true"
        is_prefill_node = self.args.run_mode == "prefill"
        is_decode_node = self.args.run_mode == "decode"
        is_nixl_prefill_node = self.args.run_mode == "nixl_prefill"
        is_nixl_decode_node = self.args.run_mode == "nixl_decode"

        if is_prefill_node:
            if self.args.dp > 1:
                self.backend = DPChunkedForPrefillNode(self.info_queue)
            else:
                self.backend = ChunckedPrefillForPrefillNode(self.info_queue)
        elif is_nixl_prefill_node:
            if self.args.dp > 1:
                self.backend = NIXLDPChunkedForPrefillNode(self.info_queue)
            else:
                self.backend = NIXLChunckedPrefillForPrefillNode(self.info_queue)

        elif is_decode_node:
            if self.args.dp > 1:
                self.backend = DPForDecodeNode(self.info_queue)
            else:
                self.backend = DecodeNode(self.info_queue)

        elif is_nixl_decode_node:
            if self.args.dp > 1:
                self.backend = NIXLDPForDecodeNode(self.info_queue)
            else:
                self.backend = NIXLDecodeNode(self.info_queue)

        elif self.args.dp > 1:
            self.backend = DPChunkedPrefillBackend()
        elif use_reward_model:
            self.backend = RewardModelBackend()
        elif return_all_prompt_logprobs:
            self.backend = ReturnPromptLogProbBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
        elif is_token_healing:
            self.backend = TokenHealingBackend()
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

        # only deepseekv3 can support auto_update_redundancy_expert
        if self.args.auto_update_redundancy_expert:
            self.redundancy_expert_manager = RedundancyExpertManager(self.backend.model)
            logger.info("init redundancy_expert_manager")
        else:
            self.redundancy_expert_manager = None
        return

    def exposed_get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()

    def release_memory_occupation(self, tags: List[MemoryTag]):
        try:
            self.backend.release_memory_occupation(tags)
            return True
        except BaseException as e:
            logger.exception(f"release memory occupation failed: {str(e)}")
            return False

    def resume_memory_occupation(self, tags: List[MemoryTag]):
        try:
            self.backend.resume_memory_occupation(tags)
            return True
        except BaseException as e:
            logger.exception(f"resume memory occupation failed: {str(e)}")
            return False

    def exposed_forward_to_model(self, req: GeneralHttpToModelRpcReq) -> GeneralModelToHttpRpcRsp:
        try:
            req = obtain(req)
            if self.backend is None or not hasattr(self.backend, req.func_name):
                raise ValueError(f"Backend does not support function {req.func_name}")
            success, ret = getattr(self.backend, req.func_name)(req.func_args)
            return GeneralModelToHttpRpcRsp(success=success, msg=str(ret), func_name=req.func_name, func_rsp=ret)
        except BaseException as e:
            logger.exception(f"forward to model backend failed: {str(e)}")
            return GeneralModelToHttpRpcRsp(
                success=False, msg=f"forward to model backend failed: {str(e)}", func_name=req.func_name
            )


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
        return

    async def init_model(self, kvargs):
        ans = self._init_model(kvargs)
        await ans
        return

    async def get_max_total_token_num(self):
        ans = self._get_max_total_token_num()
        return obtain(await ans)

    def forward_to_model(self, req: GeneralHttpToModelRpcReq) -> GeneralModelToHttpRpcRsp:
        ans = self.conn.root.forward_to_model(req)
        return obtain(ans)


def _init_env(
    args,
    rank,
    rank_in_node,
    node_world_size,
    info_queue,
    router_lock,
    socket_path,
    success_event,
):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::model_infer:RANK{rank}")
    start_parent_check_thread()

    # 将调度锁注册到全局的共享变量中
    from lightllm.common.basemodel.infer_lock import g_router_lock

    g_router_lock.obj = router_lock

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
    router_lock,
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
            router_lock,
            socket_path,
            success_event,
        ),
    )
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
