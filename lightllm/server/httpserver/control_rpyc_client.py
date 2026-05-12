import asyncio
import socket
from typing import Optional

import rpyc
from rpyc.utils.classic import obtain

from lightllm.server.io_struct import GeneralModelToHttpRpcRsp
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ControlRpycClient:
    """到 master router 控制面 rpyc service 的异步客户端。

    - 懒初始化连接, 断连时自动重连。
    - 所有调用通过 `call(method_name, *args)` 派发到 server 端的 root service。
    - 错误统一封装为 GeneralModelToHttpRpcRsp, 调用方无需处理异常。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        config: Optional[dict] = None,
        ping_timeout: float = 2.0,
    ):
        self.host = host
        self.port = port
        self.config = config if config is not None else {"allow_pickle": True, "sync_request_timeout": 600}
        self.ping_timeout = ping_timeout
        self._lock: asyncio.Lock = asyncio.Lock()
        self._conn: Optional[rpyc.core.protocol.Connection] = None

    async def _get_conn(self) -> rpyc.core.protocol.Connection:
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.ping(timeout=self.ping_timeout)
                    return self._conn
                except BaseException:
                    try:
                        self._conn.close()
                    except BaseException:
                        pass
                    self._conn = None

            self._conn = await asyncio.to_thread(
                rpyc.connect,
                self.host,
                self.port,
                config=self.config,
            )
            self._conn._channel.stream.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return self._conn

    async def call(self, method_name: str, *args) -> GeneralModelToHttpRpcRsp:
        try:
            conn = await self._get_conn()
            ret = await asyncio.to_thread(getattr(conn.root, method_name), *args)
            return obtain(ret)
        except BaseException as e:
            logger.exception(f"control rpyc call {method_name} failed: {e}")
            return GeneralModelToHttpRpcRsp(
                success=False, msg=f"control rpyc call {method_name} error: {e}", func_name=method_name
            )

    async def close(self):
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except BaseException:
                    pass
                self._conn = None
