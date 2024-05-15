import asyncio
import rpyc
from datetime import timedelta
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from lightllm.server.router.model_infer.mode_backend import (
    ContinuesBatchBackend,
    ReturnPromptLogProbBackend,
    SplitFuseBackend,
    BeamSearchBackend,
    DiversehBackend,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ModelRpcServer(rpyc.Service):
    def exposed_init_model(self, kvargs):
        self.world_size = kvargs["world_size"]
        if self.world_size != 1:
            kvargs = obtain(kvargs)
            self.world_size = kvargs["world_size"]

        is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        beam_mode = kvargs.get("beam_mode", False)
        diverse_mode = kvargs.get("diverse_mode", False)
        # use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)

        if is_splitfuse_mode:
            self.backend = SplitFuseBackend()
        elif return_all_prompt_logprobs:
            self.backend = ReturnPromptLogProbBackend()
        elif beam_mode:
            self.backend = BeamSearchBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
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
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._pause_reqs = self.model.exposed_pause_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
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


def _init_env(port):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)

    import multiprocessing

    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
