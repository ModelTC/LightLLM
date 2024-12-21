import asyncio
import rpyc
import tempfile
import torch.multiprocessing as mp
from datetime import timedelta
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from lightllm.server.router.model_infer.mode_backend import (
    ContinuesBatchBackend,
    ReturnPromptLogProbBackend,
    SplitFuseBackend,
    BeamSearchBackend,
    DiversehBackend,
    RewardModelBackend,
    TokenHealingBackend,
    SimpleConstraintBackend,
    FirstTokenConstraintBackend,
    ContinuesBatchBackendForPrefillNode,
    ContinuesBatchBackendForDecodeNode,
    DPBackend,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ModelRpcServer(rpyc.Service):
    def __init__(self, args, info_queue: mp.Queue, mem_queue: mp.Queue):
        super().__init__()
        self.args = args
        self.info_queue = info_queue
        self.mem_queue = mem_queue
        return

    def exposed_init_model(self, kvargs):
        self.world_size = kvargs["world_size"]
        if self.world_size != 1:
            kvargs = obtain(kvargs)
            self.world_size = kvargs["world_size"]

        is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        use_reward_model = kvargs.get("use_reward_model", False)
        beam_mode = kvargs.get("beam_mode", False)
        diverse_mode = kvargs.get("diverse_mode", False)
        is_token_healing = kvargs.get("is_token_healing", False)
        is_first_token_constraint_mode = kvargs.get("is_first_token_constraint_mode", False)
        if kvargs.get("args", None) is not None:
            is_simple_constraint_mode = kvargs.get("args", None).simple_constraint_mode
            is_prefill_node = kvargs.get("args", None).run_mode == "prefill"
            is_decode_node = kvargs.get("args", None).run_mode == "decode"
        else:
            is_simple_constraint_mode = False
            is_prefill_node = False
            is_decode_node = False
        # use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)
        if is_prefill_node:
            self.backend = ContinuesBatchBackendForPrefillNode(self.info_queue, self.mem_queue)
        elif is_decode_node:
            self.backend = ContinuesBatchBackendForDecodeNode(self.info_queue, self.mem_queue)
        elif use_reward_model:
            self.backend = RewardModelBackend()
        elif is_splitfuse_mode:
            self.backend = SplitFuseBackend()
        elif return_all_prompt_logprobs:
            self.backend = ReturnPromptLogProbBackend()
        elif beam_mode:
            self.backend = BeamSearchBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
        elif is_token_healing:
            self.backend = TokenHealingBackend()
        elif is_simple_constraint_mode:
            self.backend = SimpleConstraintBackend()
        elif is_first_token_constraint_mode:
            self.backend = FirstTokenConstraintBackend()
        elif kvargs.get("dp_size", 1) > 1:
            self.backend = DPBackend()
        else:
            self.backend = ContinuesBatchBackend()

        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)

        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs):
        if self.world_size != 1:
            batch_id, reqs = obtain(batch_id), obtain(reqs)

        return self.backend.add_batch(batch_id, reqs)

    # @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        if self.world_size != 1:
            batch_id = obtain(batch_id)
        return self.backend.prefill_batch(batch_id)

    # @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        if self.world_size != 1:
            batch_id = obtain(batch_id)
        return self.backend.decode_batch(batch_id)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list, finished_req_id_list = (
                obtain(batch_id),
                obtain(req_id_list),
                obtain(finished_req_id_list),
            )

        return self.backend.filter_batch(batch_id, req_id_list, finished_req_id_list)

    def exposed_pause_reqs(self, batch_id, req_list):
        if self.world_size != 1:
            batch_id, req_list = obtain(batch_id), obtain(req_list)

        return self.backend.pause_reqs(batch_id, req_list)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        if self.world_size != 1:
            batch_id1, batch_id2 = obtain(batch_id1), obtain(batch_id2)
        return self.backend.merge_batch(batch_id1, batch_id2)

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        if self.world_size != 1:
            batch_id = obtain(batch_id)

        return self.backend.remove_batch(batch_id)

    def exposed_get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:

            def async_wrap(f):
                f = rpyc.async_(f)

                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value

                return _func

            self._init_model = async_wrap(self.model.init_model)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._pause_reqs = async_wrap(self.model.pause_reqs)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
            self._get_max_total_token_num = async_wrap(self.model.get_max_total_token_num)
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._pause_reqs = self.model.exposed_pause_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
            self._get_max_total_token_num = self.model.exposed_get_max_total_token_num
        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        ans = self._filter_batch(batch_id, req_id_list, finished_req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def pause_reqs(self, batch_id, reqs_list):
        ans = self._pause_reqs(batch_id, reqs_list)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def get_max_total_token_num(self):
        ans = self._get_max_total_token_num()
        if self.use_rpc:
            return await ans
        else:
            return ans


def _init_env(args, socket_path, info_queue, mem_queue, router_lock, success_event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    # 将调度锁注册到全局的共享变量中
    from lightllm.common.basemodel.infer_lock import g_router_lock

    g_router_lock.obj = router_lock

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(
        ModelRpcServer(args, info_queue, mem_queue), socket_path=socket_path, protocol_config={"allow_pickle": True}
    )
    success_event.set()
    t.start()
    return


async def start_model_process(args, port, world_size, info_queue: mp.Queue, mem_queue: mp.Queue, router_lock: mp.Queue):
    import lightllm.utils.rpyc_fix_utils as _

    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(args, info_queue, mem_queue), world_size)

    socket_path = tempfile.mktemp()
    success_event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, socket_path, info_queue, mem_queue, router_lock, success_event))
    proc.start()
    success_event.wait(timeout=40)

    repeat_count = 0
    while repeat_count < 20:
        try:
            from rpyc.utils.factory import unix_connect

            con = unix_connect(socket_path, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
