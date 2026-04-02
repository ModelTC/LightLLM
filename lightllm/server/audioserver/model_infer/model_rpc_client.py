import asyncio
import rpyc
import threading
from typing import List
from lightllm.server.multimodal_params import AudioItem
from .model_rpc import AudioModelRpcServer


class AudioModelRpcClient:
    def __init__(self, rpc_conn, rpc_server_process=None):
        self.rpc_conn: AudioModelRpcServer = rpc_conn
        self.rpc_server_process = rpc_server_process

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                return ans.value

            return _func

        self._init_model = async_wrap(self.rpc_conn.root.init_model)
        self._run_task = async_wrap(self.rpc_conn.root.run_task)

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        await ans
        return

    async def run_task(self, audios: List[AudioItem], ref_event_list: List[threading.Event]):
        ans = self._run_task(audios, ref_event_list)
        return await ans
