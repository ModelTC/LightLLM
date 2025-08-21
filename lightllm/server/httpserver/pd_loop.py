import asyncio
import pickle
import websockets
import ujson as json
import socket
import httpx
import base64
from typing import Dict, Optional
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size
from lightllm.server.httpserver.manager import HttpServerManager
from ..pd_io_struct import PD_Master_Obj

logger = init_logger(__name__)


async def timer_log(manager: HttpServerManager):
    while True:
        await asyncio.sleep(30)
        manager.first_time_costs.print_log("mean first cost")
        manager.per_token_costs.print_log("mean per token cost")
    return


async def pd_handle_loop(manager: HttpServerManager):
    assert manager.args.host not in ["127.0.0.1", "localhost"], "pd mode must specify host ip"
    if manager.args.host in ["0.0.0.0"]:
        manager.host_ip = get_hostname_ip()
    else:
        manager.host_ip = manager.args.host

    asyncio.create_task(timer_log(manager))

    id_to_handle_task: Dict[int, asyncio.Task] = {}

    while True:
        try:
            id_to_pd_master_obj = await _get_pd_master_objs(manager.args)
            logger.info(f"get pd_master_objs {id_to_pd_master_obj}")

            if id_to_pd_master_obj is not None:
                for node_id, pd_master_obj in id_to_handle_task.items():
                    if node_id not in id_to_pd_master_obj:
                        id_to_handle_task[node_id].cancel()
                        id_to_handle_task.pop(node_id, None)
                        logger.info(f"pd_handle_task {pd_master_obj} cancelled")

                for node_id, pd_master_obj in id_to_pd_master_obj.items():
                    if node_id not in id_to_handle_task:
                        id_to_handle_task[node_id] = asyncio.create_task(_pd_handle_task(manager, pd_master_obj))

            await asyncio.sleep(30)

        except Exception as e:
            logger.exception(str(e))
            await asyncio.sleep(10)


async def _pd_handle_task(manager: HttpServerManager, pd_master_obj: PD_Master_Obj):
    """
    pd_handle_loop 主要负责与 pd master 进行注册连接，然后接收pd master发来的请求，然后
    将推理结果转发给 pd master进行处理。
    """
    # 创建转发队列
    forwarding_queue = AsyncQueue()

    while True:
        forwarding_tokens_task = None
        try:
            uri = f"ws://{pd_master_obj.host_ip_port}/pd_register"
            async with websockets.connect(
                uri, max_size=get_lightllm_websocket_max_message_size(), max_queue=(2048 * 1024, 2048 * 1023)  # 关键修改
            ) as websocket:

                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                args_dict = vars(manager.args)
                args_dict["host"] = manager.host_ip
                # 发送注册信息
                regist_json = {
                    "node_id": manager.args.pd_node_id,
                    "client_ip_port": f"{manager.host_ip}:{manager.args.port}",
                    "mode": manager.pd_mode.value,
                    "start_args": args_dict,
                }

                await websocket.send(json.dumps(regist_json))
                logger.info(f"Sent registration JSON: {regist_json}")

                # 转发任务
                forwarding_tokens_task = asyncio.create_task(_up_tokens_to_pd_master(forwarding_queue, websocket))

                # 接收 pd master 发来的请求，并推理后，将生成的token转发回pd master。
                while True:
                    recv_bytes = await websocket.recv()
                    obj = pickle.loads(recv_bytes)
                    if obj[0] == ObjType.REQ:
                        prompt, sampling_params, multimodal_params = obj[1]
                        asyncio.create_task(
                            _pd_process_generate(manager, prompt, sampling_params, multimodal_params, forwarding_queue)
                        )
                    elif obj[0] == ObjType.ABORT:
                        group_req_id = obj[1]
                        await manager.abort(group_req_id)
                    else:
                        logger.error(f"recevie error obj {str(obj)}")

        except asyncio.CancelledError:
            # 如果任务被取消，则退出循环
            logger.warning(f"forwarding_tokens_task {pd_master_obj} cancelled")
            if forwarding_tokens_task is not None:
                forwarding_tokens_task.cancel()
            return

        except Exception as e:
            logger.error("connetion to pd_master has error")
            logger.exception(str(e))
            if forwarding_tokens_task is not None:
                forwarding_tokens_task.cancel()
            await asyncio.sleep(10)
            await forwarding_queue.get_all_data()
            logger.info("reconnection to pd_master")


async def _get_pd_master_objs(args) -> Optional[Dict[int, PD_Master_Obj]]:
    """
    get_pd_master_objs 主要负责从 pd master 获取所有的pd master对象。
    """
    use_config_server = args.config_server_host and args.config_server_port

    # 如果不使用config_server服务来发现所有的 pd_master, 则需要使用启动参数中的
    # --pd_master_ip 和--pd_master_port 设置的唯一pd_master来进行连接, 其默认
    # node_id 为 0
    if not use_config_server:
        ans = dict()
        ans[0] = PD_Master_Obj(node_id=0, host_ip_port=f"{args.pd_master_ip}:{args.pd_master_port}")
        return ans

    # 使用 config_server 服务来发现所有的 pd_master 节点。
    uri = f"ws://{args.config_server_host}:{args.config_server_port}/registered_objects"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(uri)
            if response.status_code == 200:
                base64data = response.json()["data"]
                id_to_pd_master_obj = pickle.loads(base64.b64decode(base64data))
                return id_to_pd_master_obj
            else:
                logger.error(f"get pd_master_objs error {response.status_code}")
                return None
    except Exception as e:
        logger.exception(str(e))
        await asyncio.sleep(10)
        return None


# 触发推理的task
async def _pd_process_generate(
    manager: HttpServerManager, prompt, sampling_params, multimodal_params, forwarding_queue: AsyncQueue
):
    try:
        async for sub_req_id, request_output, metadata, finish_status in manager.generate(
            prompt, sampling_params, multimodal_params, None
        ):
            # p d 模式下，将 token 数据放入到转发队列中, 请求id 小于0的请求是health探测请求，不用转发。
            is_health_check_req = sub_req_id < 0
            if not is_health_check_req:
                await forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))

    except BaseException as e:
        logger.error(str(e))


# 转发token的task
async def _up_tokens_to_pd_master(forwarding_queue: AsyncQueue, websocket):
    while True:
        handle_list = await forwarding_queue.wait_to_get_all_data()

        if handle_list:
            load_info: dict = _get_load_info()
            await websocket.send(pickle.dumps((ObjType.TOKEN_PACKS, handle_list, load_info)))


# 获取节点负载信息
def _get_load_info() -> dict:

    from lightllm.server.api_http import g_objs

    assert g_objs.shared_token_load is not None, "shared_token_load is not initialized"
    args = g_objs.args
    dp_size_in_node = max(1, args.dp // args.nnodes)

    # 获取当前每个 dp 的负载，数值含义为当前的 token 总容量使用率， 上报给 PD_Master 用于做
    # 调度决策。
    current_load = [
        float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(dp_size_in_node)
    ]
    mean_node_load = sum(current_load) / len(current_load)
    load_info = {
        "total_token_usage_rate": mean_node_load,
        "client_ip_port": f"{g_objs.httpserver_manager.host_ip}:{g_objs.args.port}",
    }
    return load_info
