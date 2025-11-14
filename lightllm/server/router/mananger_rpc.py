import rpyc
import asyncio
import socket
from .manager import RouterManager


class RouterRpcService(rpyc.Service):
    def __init__(self, router_manager: "RouterManager"):
        super().__init__()
        self.router_manager = router_manager
        return

    def exposed_flush_cache(self) -> bool:
        return self.router_manager.flush_cache()


class RouterRpcClient:
    def __init__(self, router_rpc_conn):
        self.router_rpc_conn = router_rpc_conn

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                # raise if exception
                return ans.value

            return _func

        self._flush_cache = async_wrap(self.router_rpc_conn.root.flush_cache)
        return

    async def flush_cache(self) -> bool:
        ans = await self._flush_cache()
        return ans


def connect_router_rpc(port: int) -> RouterRpcClient:
    router_rpc_conn = rpyc.connect("localhost", port, config={"allow_pickle": True})
    router_rpc_conn._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return RouterRpcClient(router_rpc_conn)
