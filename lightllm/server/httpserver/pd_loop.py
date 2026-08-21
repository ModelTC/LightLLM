import asyncio
import pickle
import websockets
import ujson as json
import socket
import httpx
import base64
import weakref
import os
import signal
import sys
from typing import Dict, Optional, Union, List
from websockets import ClientConnection
from lightllm.server.pd_io_struct import (
    NodeRole,
    ObjType,
    PD_COMPACT_TOKEN_INFO_LEN,
    build_pd_compact_token_info,
)
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size
from lightllm.server.httpserver.manager import HttpServerManager
from ..pd_io_struct import PD_Master_Obj
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs import SamplingParams
from lightllm.utils.error_utils import PDPrefillNodeStopGenToken
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)


async def timer_log(manager: HttpServerManager):
    while True:
        await asyncio.sleep(30)
        manager.first_time_costs.print_log("mean first cost")
        manager.per_token_costs.print_log("mean per token cost")
    return


async def pd_handle_loop(manager: HttpServerManager):
    if manager.args.host in ["127.0.0.1", "localhost"]:
        logger.error("pd mode must specify host ip, not use 127.0.0.1 or localhost")
        # kill father process to trigger graceful exit, avoid orphan process
        os.kill(os.getppid(), signal.SIGINT)
        sys.exit(-1)

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
        heartbeat_task = None
        generation_tasks: Dict[int, asyncio.Task] = {}
        try:
            uri = f"ws://{pd_master_obj.host_ip_port}/pd_register"
            async with websockets.connect(
                uri,
                max_size=get_lightllm_websocket_max_message_size(),
                max_queue=(2048 * 1024, 2048 * 1023),  # 关键修改
                # 下方应用层心跳已负责存活检测，禁用协议层 keepalive，避免繁忙连接被误断。
                ping_interval=None,
            ) as websocket:

                sock = websocket.transport.get_extra_info("socket")
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                args_dict = vars(manager.args)
                args_dict["host"] = manager.host_ip
                # 发送注册信息
                regist_json = {
                    "node_id": manager.args.pd_node_id,
                    "client_ip_port": f"{manager.host_ip}:{get_shm_port_args().port}",
                    "mode": manager.pd_mode.value,
                    "start_args": args_dict,
                }

                await websocket.send(json.dumps(regist_json))
                logger.info(f"Sent registration JSON: {regist_json}")

                # 转发任务
                forwarding_tokens_task = asyncio.create_task(_up_tokens_to_pd_master(forwarding_queue, websocket))
                heartbeat_task = asyncio.create_task(_send_heartbeat_to_pd_master(websocket))

                group_req_id_to_event: Dict[int, asyncio.Event] = weakref.WeakValueDictionary()
                # 接收 pd master 发来的请求，并推理后，将生成的token转发回pd master。
                while True:
                    recv_bytes = await websocket.recv()
                    obj = pickle.loads(recv_bytes)
                    if obj[0] == ObjType.REQ:
                        prompt, sampling_params, multimodal_params = obj[1]
                        group_req_id = sampling_params.group_request_id
                        pd_event = asyncio.Event()
                        group_req_id_to_event[group_req_id] = pd_event
                        manager.begin_pd_request_registration(group_req_id)
                        generation_task = asyncio.create_task(
                            _pd_process_generate(
                                manager=manager,
                                prompt=prompt,
                                sampling_params=sampling_params,
                                multimodal_params=multimodal_params,
                                forwarding_queue=forwarding_queue,
                                pd_upload_websocket=websocket,
                                pd_event=pd_event,
                            )
                        )
                        generation_tasks[group_req_id] = generation_task

                        def remove_generation_task(task: asyncio.Task, request_id: int = group_req_id):
                            if generation_tasks.get(request_id) is task:
                                generation_tasks.pop(request_id, None)
                                # task 可能在首次运行前被取消，此时协程内的 finally 不会执行。
                                manager.cancel_pd_request_registration(request_id)

                        generation_task.add_done_callback(remove_generation_task)
                    elif obj[0] == ObjType.ABORT:
                        group_req_id = obj[1]
                        logger.warning(f"recv cmd aborted req id {group_req_id}")
                        generation_task = generation_tasks.get(group_req_id)
                        if generation_task is not None and not generation_task.done():
                            generation_task.cancel()
                        await manager.abort(group_req_id)

                    elif obj[0] == ObjType.PD_REQ_DECODE_NODE_INFO:
                        _, group_req_id, decode_node_info = obj
                        pd_event = group_req_id_to_event.pop(group_req_id, None)
                        if pd_event is None:
                            logger.error(f"error in find pd_event, info: {obj}")
                            continue
                        pd_event.decode_node_info = decode_node_info
                        pd_event.set()
                    else:
                        logger.error(f"recevie error obj {str(obj)}")

        except asyncio.CancelledError:
            # 如果任务被取消，则退出循环
            logger.warning(f"pd_handle_task {pd_master_obj} cancelled")
            return

        except Exception as e:
            logger.error("connetion to pd_master has error")
            logger.exception(str(e))
        finally:
            child_tasks = [task for task in (forwarding_tokens_task, heartbeat_task) if task is not None]
            child_tasks.extend(generation_tasks.values())
            for task in child_tasks:
                task.cancel()
            if child_tasks:
                await asyncio.gather(*child_tasks, return_exceptions=True)

        await asyncio.sleep(10)
        await forwarding_queue.get_all_data()
        logger.info("reconnection to pd_master")


async def _get_pd_master_objs(args: StartArgs) -> Optional[Dict[int, PD_Master_Obj]]:
    """
    get_pd_master_objs 主要负责从 pd master 获取所有的pd master对象。
    """
    use_config_server = args.config_server_host and args.config_server_port

    # 如果不使用config_server服务来发现所有的 pd_master, 则需要使用启动参数中的
    # --pd_master_ip 和--pd_master_port 设置的唯一pd_master来进行连接, 其默认
    # node_id 为 0
    if not use_config_server:
        ans = dict()
        ans[0] = PD_Master_Obj(node_id=0, host_ip_port=f"{args.pd_master_ip}:{get_shm_port_args().pd_master_port}")
        return ans

    # 使用 config_server 服务来发现所有的 pd_master 节点。
    uri = f"ws://{args.config_server_host}:{get_shm_port_args().config_server_port}/registered_objects"

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
    manager: HttpServerManager,
    prompt: Union[str, List[int]],
    sampling_params: SamplingParams,
    multimodal_params: Dict,
    forwarding_queue: AsyncQueue,
    pd_upload_websocket: ClientConnection,
    pd_event: asyncio.Event,
):
    return_output_logprobs = getattr(sampling_params, "return_output_logprobs", True)
    try:
        async for sub_req_id, request_output, metadata, finish_status in manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            pd_upload_websocket=pd_upload_websocket,
            pd_event=pd_event,
        ):
            metadata.pop("prompt_ids", None)
            if not return_output_logprobs:
                for key in ("id", "logprob", "cumlogprob", "special", "logprobs"):
                    metadata.pop(key, None)
            if metadata.get("count_output_tokens") == 1:
                metadata["node_mode"] = manager.args.run_mode
            if not return_output_logprobs:
                compact_token_info = build_pd_compact_token_info(sub_req_id, request_output, metadata, finish_status)
                if compact_token_info is not None:
                    await forwarding_queue.put(compact_token_info)
                    continue
            await forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))
    except PDPrefillNodeStopGenToken as e:
        logger.info(f"pd prefill node stop gen token for group_request_id {e.group_request_id}")
    except BaseException as e:
        logger.error(str(e))
    finally:
        manager.cancel_pd_request_registration(sampling_params.group_request_id)


# 转发token的task
async def _up_tokens_to_pd_master(forwarding_queue: AsyncQueue, websocket: ClientConnection):
    max_message_size = get_lightllm_websocket_max_message_size()

    while True:
        handle_list = await forwarding_queue.wait_to_get_all_data()

        if handle_list:
            load_info: dict = _get_load_info()
            pending_handle_lists = []
            group_start = 0
            group_is_compact = len(handle_list[0]) == PD_COMPACT_TOKEN_INFO_LEN
            for index in range(1, len(handle_list)):
                item_is_compact = len(handle_list[index]) == PD_COMPACT_TOKEN_INFO_LEN
                if item_is_compact != group_is_compact:
                    pending_handle_lists.append(handle_list[group_start:index])
                    group_start = index
                    group_is_compact = item_is_compact
            pending_handle_lists.append(handle_list[group_start:])
            pending_handle_lists.reverse()
            while pending_handle_lists:
                token_list = pending_handle_lists.pop()
                obj_type = (
                    ObjType.TOKEN_PACKS_COMPACT
                    if len(token_list[0]) == PD_COMPACT_TOKEN_INFO_LEN
                    else ObjType.TOKEN_PACKS
                )
                payload = pickle.dumps((obj_type, token_list, load_info))
                if len(payload) <= max_message_size:
                    await websocket.send(payload)
                    continue

                if len(token_list) == 1:
                    raise ValueError(
                        f"single PD token pack is {len(payload)} bytes, exceeding websocket limit "
                        f"{max_message_size}"
                    )

                split_index = len(token_list) // 2
                logger.warning(
                    f"PD token pack is {len(payload)} bytes with {len(token_list)} items, exceeding websocket "
                    f"limit {max_message_size}; splitting it"
                )
                # 栈后进先出，先压后半段，保持 token 的原始发送顺序。
                pending_handle_lists.extend((token_list[split_index:], token_list[:split_index]))


async def _send_heartbeat_to_pd_master(websocket: ClientConnection):
    heartbeat_interval_seconds = 15
    while True:
        await websocket.send(pickle.dumps((ObjType.HEARTBEAT,)))
        await asyncio.sleep(heartbeat_interval_seconds)


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
        "client_ip_port": f"{g_objs.httpserver_manager.host_ip}:{get_shm_port_args().port}",
    }
    return load_info
