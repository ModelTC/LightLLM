import asyncio
import pickle
import websockets
import socket
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from ..pd_io_struct import PD_Master_Obj

logger = init_logger(__name__)


async def register_loop(manager: HttpServerManagerForPDMaster):
    assert manager.args.host not in ["127.0.0.1", "localhost"], "pd mode must specify host ip"

    if manager.args.host in ["0.0.0.0"]:
        manager.host_ip = get_hostname_ip()
    else:
        manager.host_ip = manager.args.host

    while True:

        try:
            uri = f"ws://{manager.args.config_server_host}:{manager.args.config_server_port}/pd_master_register"
            async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:

                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                pd_master_obj = PD_Master_Obj(
                    node_id=manager.args.pd_node_id, host_ip_port=f"{manager.host_ip}:{manager.args.port}"
                )

                await websocket.send(pickle.dumps(pd_master_obj))
                logger.info(f"Sent registration pd_master obj: {pd_master_obj}")

                while True:
                    await websocket.send("heartbeat")
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error("connetion to config_server has error")
            logger.exception(str(e))
            await asyncio.sleep(10)
            logger.info("reconnection to config_server")
