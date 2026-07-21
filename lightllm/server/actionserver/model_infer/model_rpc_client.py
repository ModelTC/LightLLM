import asyncio

import rpyc
from rpyc.utils.classic import obtain


class ActionModelRpcClient:
    def __init__(self, rpc_conn):
        self.rpc_conn = rpc_conn

        def async_wrap(func):
            remote = rpyc.async_(func)

            async def wrapped(*args, **kwargs):
                result = remote(*args, **kwargs)
                await asyncio.to_thread(result.wait)
                return obtain(result.value)

            return wrapped

        self._init_model = async_wrap(self.rpc_conn.root.init_model)
        self._run_task = async_wrap(self.rpc_conn.root.run_task)
        self._release_prefix_context = async_wrap(self.rpc_conn.root.release_prefix_context)

    async def init_model(self, kvargs):
        return await self._init_model(kvargs)

    async def run_task(self, task):
        return await self._run_task(task)

    async def release_prefix_context(self, identity):
        return await self._release_prefix_context(identity)
