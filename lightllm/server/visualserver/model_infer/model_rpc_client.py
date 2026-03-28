import asyncio
import rpyc
import threading
from typing import Dict, List, Tuple, Deque, Optional, Union
from lightllm.server.multimodal_params import ImageItem
from .model_rpc import VisualModelRpcServer
from .visual_only_model_rpc import VisualOnlyModelRpcServer
from lightllm.utils.envs_utils import get_env_start_args


class VisualModelRpcClient:
    def __init__(self, rpc_conn):
        self.rpc_conn: Union[VisualModelRpcServer, VisualOnlyModelRpcServer] = rpc_conn

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                # raise if exception
                return ans.value

            return _func

        self._init_model = async_wrap(self.rpc_conn.root.init_model)
        self._encode = async_wrap(self.rpc_conn.root.encode)
        if get_env_start_args().run_mode == "visual_only":
            self._run_task = async_wrap(self.rpc_conn.root.run_task)

        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        await ans
        return

    async def encode(self, images: List[ImageItem]):
        ans = self._encode(images)
        return await ans
    
    async def run_task(self, images: List[ImageItem], ref_event: threading.Event):
        ans = self._run_task(images, ref_event)
        return await ans
