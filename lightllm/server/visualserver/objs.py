import asyncio
from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

rpyc_config = {
    "allow_pickle": True,
    "allow_all_attrs": True,
    "allow_getattr": True,
    "allow_setattr": True,
}


class ImageReadyEvent:
    """Let an RPyC worker notify the visual manager's event loop."""

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()

    def set(self):
        self._loop.call_soon_threadsafe(self._ready.set)

    async def wait(self):
        await self._ready.wait()


@dataclass
class VIT_Obj:
    node_id: int
    host_ip: str
    port: int

    def to_log_str(self):
        return f"VIT host_ip_port: {self.host_ip}:{self.port}, node_id: {self.node_id}"
