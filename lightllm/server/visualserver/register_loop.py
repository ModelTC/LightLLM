import asyncio
import pickle
import websockets
import socket
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from .vit_connect import VIT_Obj

logger = init_logger(__name__)


async def register_loop(args):
    assert args.host not in ["127.0.0.1", "localhost"], "remote visual server must specify host ip"

    if args.host in ["0.0.0.0"]:
        host_ip = get_hostname_ip()
    else:
        host_ip = args.host

    while True:

        try:
            uri = f"ws://{args.config_server_host}:{args.config_server_port}/visual_register"
            async with websockets.connect(uri, max_queue=(2048 * 1024, 2048 * 1023)) as websocket:

                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                vit_obj = VIT_Obj(node_id=args.visual_node_id, host_ip_port=f"{host_ip}:{args.port}")

                await websocket.send(pickle.dumps(vit_obj))
                logger.info(f"Sent registration vit_obj: {vit_obj}")

                while True:
                    await websocket.send("heartbeat")
                    await asyncio.sleep(40)

        except Exception as e:
            logger.error("connetion to config_server has error")
            logger.exception(str(e))
            await asyncio.sleep(10)
            logger.info("reconnection to config_server")
