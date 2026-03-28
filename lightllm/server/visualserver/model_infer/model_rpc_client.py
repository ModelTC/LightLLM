import asyncio
import rpyc
import threading
from typing import Dict, List, Tuple, Deque, Optional, Union
from lightllm.server.multimodal_params import ImageItem
from .model_rpc import VisualModelRpcServer
from .visual_only_model_rpc import VisualOnlyModelRpcServer


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
        self._run_task = async_wrap(self.rpc_conn.root.run_task)

        return

    async def init_model(self, kvargs):
        ans: rpyc.AsyncResult = self._init_model(kvargs)
        await ans
        return
    
    async def run_task(self, images: List[ImageItem], ref_event_list: List[threading.Event]):
        ans = self._run_task(images, ref_event_list)
        return await ans
