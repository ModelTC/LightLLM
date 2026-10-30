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
import uuid
from typing import Dict, Optional, Union, List
from websockets import ClientConnection
from lightllm.server.pd_io_struct import NodeRole, ObjType
from lightllm.server.httpserver.async_queue import AsyncQueue
from lightllm.utils.net_utils import get_hostname_ip, format_host_port
from lightllm.utils.log_utils import (
    init_logger,
    set_http_request_id,
    reset_http_request_id,
    log_debug_event,
    log_repeated_exception,
)
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size
from lightllm.server.httpserver.manager import HttpServerManager
from ..pd_io_struct import PD_Client_Obj, PD_Master_Obj
from lightllm.server.core.objs import StartArgs
from lightllm.server.core.objs import SamplingParams
from lightllm.utils.error_utils import PDPrefillNodeStopGenToken, ServerBusyError
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)


async def timer_log(manager: HttpServerManager):
    while True:
        await asyncio.sleep(30)
        manager.first_time_costs.print_log("mean first cost")
        manager.per_token_costs.print_log("mean per token cost")
    return


async def pd_handle_loop(manager: HttpServerManager, dp_index: Optional[int] = None):
    if manager.args.host in ["127.0.0.1", "localhost"]:
        logger.error("pd mode must specify host ip, not use 127.0.0.1 or localhost")
        # kill father process to trigger graceful exit, avoid orphan process
        os.kill(os.getppid(), signal.SIGINT)
        sys.exit(-1)

    if manager.args.host in ["0.0.0.0"]:
        manager.host_ip = get_hostname_ip()
    else:
        manager.host_ip = manager.args.host

    id_to_handle_task: Dict[int, asyncio.Task] = {}
    previous_master_ids = object()

    while True:
        try:
            id_to_pd_master_obj = await _get_pd_master_objs(manager.args)
            master_ids = tuple(sorted(id_to_pd_master_obj or {}))
            if master_ids != previous_master_ids:
                event = "pd.masters_changed" if master_ids else "pd.waiting_for_master"
                logger.info(
                    "event=%s master_ids=%s", event, master_ids, extra={"event": event, "master_ids": master_ids}
                )
                previous_master_ids = master_ids

            if id_to_pd_master_obj is not None:
                for node_id, pd_master_obj in list(id_to_handle_task.items()):
                    if node_id not in id_to_pd_master_obj:
                        id_to_handle_task[node_id].cancel()
                        id_to_handle_task.pop(node_id, None)
                        logger.info(f"pd_handle_task {pd_master_obj} cancelled")

                for node_id, pd_master_obj in id_to_pd_master_obj.items():
                    if node_id not in id_to_handle_task:
                        id_to_handle_task[node_id] = asyncio.create_task(
                            _pd_handle_task(manager, pd_master_obj, dp_index)
                        )

            await asyncio.sleep(30)

        except Exception as e:
            logger.exception(str(e))
            await asyncio.sleep(10)


async def _pd_handle_task(manager: HttpServerManager, pd_master_obj: PD_Master_Obj, dp_index: Optional[int] = None):
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

                args_dict = vars(manager.args).copy()
                args_dict["host"] = manager.host_ip
                # 发送注册信息
                regist_json = {
                    "node_id": manager.args.pd_node_id,
                    "client_ip_port": format_host_port(manager.host_ip, get_shm_port_args().port),
                    "mode": manager.pd_mode.value,
                    "start_args": args_dict,
                    "dp_index": dp_index,
                    "connection_id": uuid.uuid4().hex,
                }

                await websocket.send(json.dumps(regist_json))
                logger.info(
                    "event=pd.registration_sent role=%s endpoint=%s dp_index=%s",
                    regist_json["mode"],
                    regist_json["client_ip_port"],
                    dp_index,
                    extra={"event": "pd.registration_sent"},
                )

                # 转发任务
                forwarding_tokens_task = asyncio.create_task(
                    _up_tokens_to_pd_master(forwarding_queue, websocket, dp_index)
                )
                heartbeat_task = asyncio.create_task(_send_heartbeat_to_pd_master(websocket))

                group_req_id_to_event: Dict[int, asyncio.Event] = weakref.WeakValueDictionary()
                # 接收 pd master 发来的请求，并推理后，将生成的token转发回pd master。
                while True:
                    recv_bytes = await websocket.recv()
                    obj = pickle.loads(recv_bytes)
                    if obj[0] == ObjType.REQ:
                        prompt, sampling_params, multimodal_params = obj[1]
                        log_metadata = obj[2] if len(obj) > 2 else {}
                        if dp_index is not None:
                            if sampling_params.suggested_dp_index != -1:
                                logger.warning(
                                    f"group_request_id={sampling_params.group_request_id}: "
                                    f"suggested_dp_index={sampling_params.suggested_dp_index} is unexpected "
                                    f"in DP split connection mode; overriding with connection dp_index={dp_index}"
                                )
                            # 连接对应固定的本地 DP rank，master 无需感知请求的 rank 分配。
                            sampling_params.suggested_dp_index = dp_index
                        group_req_id = sampling_params.group_request_id
                        pd_event = asyncio.Event()
                        group_req_id_to_event[group_req_id] = pd_event
                        generation_task = asyncio.create_task(
                            _pd_process_generate(
                                manager=manager,
                                prompt=prompt,
                                sampling_params=sampling_params,
                                multimodal_params=multimodal_params,
                                forwarding_queue=forwarding_queue,
                                pd_upload_websocket=websocket,
                                pd_event=pd_event,
                                http_request_id=log_metadata.get("http_request_id"),
                            )
                        )
                        generation_tasks[group_req_id] = generation_task

                        def remove_generation_task(task: asyncio.Task, request_id: int = group_req_id):
                            if generation_tasks.get(request_id) is task:
                                generation_tasks.pop(request_id, None)

                        generation_task.add_done_callback(remove_generation_task)
                    elif obj[0] == ObjType.ABORT:
                        group_req_id = obj[1]
                        logger.warning(f"recv cmd aborted req id {group_req_id}")
                        generation_task = generation_tasks.get(group_req_id)
                        if generation_task is not None and not generation_task.done():
                            generation_task.cancel()
                        if not (await manager.abort(group_req_id)):

                            async def delayed_abort_task(group_req_id, retry_count):
                                for _ in range(retry_count):
                                    await asyncio.sleep(5.0)
                                    if await manager.abort(group_req_id):
                                        break

                            asyncio.create_task(delayed_abort_task(group_req_id=group_req_id, retry_count=4))

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

        except Exception:
            log_repeated_exception(logger, "pd.connection_failed", master=pd_master_obj.host_ip_port)
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
    """通过 config server 发现节点，未配置时查询指定 master 的 worker 端口。"""
    use_config_server = args.config_server_host is not None and args.config_server_port is not None
    ports = get_shm_port_args()

    async def get_from_config_server(client, endpoint):
        response = await client.get(f"http://{endpoint}/registered_objects")
        if response.status_code != 200:
            logger.error(f"get pd_master_objs error {response.status_code}")
            return None
        base64data = response.json()["data"]
        return pickle.loads(base64.b64decode(base64data))

    async def get_from_pd_master(client):
        endpoint = format_host_port(args.pd_master_ip, ports.pd_master_port)
        response = await client.get(f"http://{endpoint}/get_server_info")
        response.raise_for_status()
        info = response.json()
        if info["run_mode"] != "pd_master":
            raise ValueError(f"{endpoint} is not a PD master")

        worker_count = info["httpserver_workers"]
        if worker_count == 1:
            # 单 worker 可能通过端口映射或代理对外暴露。保留已经成功访问的
            # endpoint，只从响应中读取节点 ID，避免改连进程内部监听端口。
            node_id = info["pd_node_id"]
            return {node_id: PD_Master_Obj(node_id=node_id, host_ip_port=endpoint)}

        if worker_count > 1:
            # 本机多 worker 使用监听端口作为节点 ID，可由端口列表直接构建连接表。
            return {
                worker_port: PD_Master_Obj(
                    node_id=worker_port,
                    host_ip_port=format_host_port(args.pd_master_ip, worker_port),
                )
                for worker_port in info["pd_master_multi_ports"]
            }

        raise ValueError(f"{endpoint} returned invalid httpserver_workers={worker_count}")

    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            if use_config_server:
                endpoint = format_host_port(args.config_server_host, ports.config_server_port)
                return await get_from_config_server(client, endpoint)
            else:
                return await get_from_pd_master(client)
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
    http_request_id: Optional[str] = None,
):
    context_token = set_http_request_id(http_request_id or "-")
    try:
        async for sub_req_id, request_output, metadata, finish_status in manager.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            multimodal_params=multimodal_params,
            request=None,
            pd_upload_websocket=pd_upload_websocket,
            pd_event=pd_event,
        ):
            metadata["node_mode"] = manager.args.run_mode
            await forwarding_queue.put((sub_req_id, request_output, metadata, finish_status))
    except PDPrefillNodeStopGenToken as e:
        logger.debug(f"pd prefill node stop gen token for group_request_id {e.group_request_id}")
    except ServerBusyError as e:
        group_request_id = sampling_params.group_request_id
        logger.warning(f"pd node rejected request {group_request_id}: {e.message}")
        try:
            await pd_upload_websocket.send(pickle.dumps((ObjType.PD_UPLOAD_SERVER_BUSY, group_request_id, e.message)))
        except Exception:
            logger.exception(f"report pd node request rejection failed, group_request_id: {group_request_id}")
    except asyncio.CancelledError:
        # PD master 主动 abort 或连接断开清理任务时会走取消路径，不需要反向重复上报。
        pass
    except BaseException as e:
        group_request_id = sampling_params.group_request_id
        logger.exception(f"pd node generate request {group_request_id} failed: {str(e)}")
        try:
            # 本地生成在任意阶段失败后，及时通知 PD master 终止对应请求，避免 master
            # 只能依赖 prefill/decode 阶段的超时才能发现异常。
            await pd_upload_websocket.send(
                pickle.dumps((ObjType.PD_UPLOAD_GENERATE_ERROR, group_request_id, f"{type(e).__name__}: {str(e)}"))
            )
        except Exception:
            logger.exception(f"report pd node generate error failed, group_request_id: {group_request_id}")
    finally:
        reset_http_request_id(context_token)


# 转发token的task
async def _up_tokens_to_pd_master(
    forwarding_queue: AsyncQueue, websocket: ClientConnection, dp_index: Optional[int] = None
):
    while True:
        handle_list = await forwarding_queue.wait_to_get_all_data()

        if handle_list:
            load_info: dict = _get_load_info(dp_index)
            await websocket.send(pickle.dumps((ObjType.TOKEN_PACKS, handle_list, load_info)))


async def _send_heartbeat_to_pd_master(websocket: ClientConnection):
    heartbeat_interval_seconds = 15
    while True:
        await websocket.send(pickle.dumps((ObjType.HEARTBEAT,)))
        await asyncio.sleep(heartbeat_interval_seconds)


# 获取节点负载信息
def _get_load_info(dp_index: Optional[int] = None) -> dict:
    from lightllm.server.api_http import g_objs

    assert g_objs.shared_token_load is not None, "shared_token_load is not initialized"
    args = g_objs.args
    dp_size_in_node = max(1, args.dp // args.nnodes)

    # 获取当前每个 dp 的负载，数值含义为当前的 token 总容量使用率， 上报给 PD_Master 用于做
    # 调度决策。
    current_load = [
        float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(dp_size_in_node)
    ]
    # 拆分连接只上报对应 rank 的负载；普通连接仍上报节点平均负载。
    mean_node_load = sum(current_load) / len(current_load) if dp_index is None else current_load[dp_index]
    load_info = {
        "total_token_usage_rate": mean_node_load,
        "connection_key": PD_Client_Obj.make_connection_key(
            format_host_port(g_objs.httpserver_manager.host_ip, get_shm_port_args().port), dp_index
        ),
    }
    return load_info
