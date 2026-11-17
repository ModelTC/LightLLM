"""PD 节点上，本机进程到 httpserver 的多对一通道。

多个进程各自 PUSH，一个 httpserver worker PULL。消息是 pickle 后的对象，
通道本身不限定类型。当前 decode 传输进程用它把 PDUpKVStatus 交给 httpserver，
再由 httpserver 转发到 PD Master。以后同一条通道也可以传其他对象。

httpserver 先 bind，同一个地址只能绑一次，失败直接抛异常。
传输进程在 httpserver 起来之后再 connect。连接保持时，各发送方的消息都会送到，
同一发送方内部保持顺序。send() 返回只表示进入本进程队列，对端没有确认。
对端还没接上、连接断开或进程退出时，队列里的消息会丢。
"""

import pickle

from lightllm.utils.net_utils import get_pd_httpserver_ipc_address


class PDHttpserverIpc:
    class Push:
        def __init__(self) -> None:
            self._context = None
            self._socket = None

        def send(self, message: object) -> None:
            # 第一次发送时才 connect。返回只表示消息进了本进程队列。
            self._ensure_connected()
            self._socket.send(pickle.dumps(message))

        def _ensure_connected(self) -> None:
            if self._socket is not None:
                return
            import zmq

            self._context = zmq.Context(1)
            socket = self._context.socket(zmq.PUSH)
            socket.connect(get_pd_httpserver_ipc_address())
            self._socket = socket

    class Pull:
        def __init__(self) -> None:
            self._context = None
            self._socket = None

        def bind(self) -> None:
            # httpserver 启动时绑定。地址已被占用则直接抛异常。
            import zmq.asyncio

            self._context = zmq.asyncio.Context(1)
            self._socket = self._context.socket(zmq.PULL)
            self._socket.bind(get_pd_httpserver_ipc_address())

        async def recv(self) -> object:
            # 一直等到下一条本机消息，再还原成发送时的对象。
            payload = await self._socket.recv()
            return pickle.loads(payload)

        def close(self) -> None:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
