"""本机到 pd_report 的 ZMQ PUSH/PULL。失败时退出进程。"""

import asyncio
import os
import pickle

from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_pd_report_ipc_address

logger = init_logger(__name__)


class ReportIpc:
    class Push:
        def __init__(self) -> None:
            self._context = None
            self._socket = None
            self._address = get_pd_report_ipc_address()

        def send(self, message: object) -> None:
            try:
                self._ensure_connected()
                self._socket.send(pickle.dumps(message))
            except Exception as exc:
                _abort(action="send", address=self._address, exc=exc)

        def close(self) -> None:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None

        def _ensure_connected(self) -> None:
            if self._socket is not None:
                return
            import zmq

            self._context = zmq.Context(1)
            sock = self._context.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._address)
            self._socket = sock

    class Pull:
        def __init__(self, address: str) -> None:
            self._context = None
            self._socket = None
            self._address = address

        def bind(self) -> None:
            import zmq.asyncio

            self._context = zmq.asyncio.Context(1)
            self._socket = self._context.socket(zmq.PULL)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.bind(self._address)

        async def recv(self) -> object:
            try:
                payload = await self._socket.recv()
                return pickle.loads(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _abort(action="recv", address=self._address, exc=exc)

        def close(self) -> None:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None


def _abort(action: str, address: str, exc: BaseException) -> None:
    logger.exception(f"pd report ipc {action} failed, address={address}: {exc}")
    os._exit(1)
