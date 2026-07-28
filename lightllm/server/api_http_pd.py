"""PD separation control-plane WebSocket APIs.

供 prefill / decode 节点与 pd_master 通信：
  - ``/pd_register``：P/D 节点注册与请求转发
  - ``/kv_move_status``：decode 节点上报 KV 传输状态

路由在模块级 ``router`` 上注册，由 ``api_http`` ``include_router`` 挂载。
``g_objs`` 在 handler 内懒导入，避免与 api_http 循环依赖。
"""

import pickle

import ujson as json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.websocket("/pd_register")
async def register_and_keep_alive(websocket: WebSocket):
    from .api_http import g_objs

    await websocket.accept()
    websocket._receive_bytes_max_size = get_lightllm_websocket_max_message_size()
    client_ip, client_port = websocket.client
    logger.info(f"Client connected from IP: {client_ip}, Port: {client_port}")
    regist_json = json.loads(await websocket.receive_text())
    logger.info(f"received regist_json {regist_json}")
    await g_objs.httpserver_manager.register_pd(regist_json, websocket)

    try:
        while True:
            data = await websocket.receive_bytes()
            obj = pickle.loads(data)
            await g_objs.httpserver_manager.put_to_handle_queue(obj)

    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"client {regist_json} has error {str(e)}")
        logger.exception(str(e))
    finally:
        logger.error(f"client {regist_json} removed")
        await g_objs.httpserver_manager.remove_pd(regist_json)
    return


@router.websocket("/kv_move_status")
async def kv_move_status(websocket: WebSocket):
    from .api_http import g_objs

    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"kv_move_status Client connected from IP: {client_ip}, Port: {client_port}")
    try:
        while True:
            data = await websocket.receive_bytes()
            upkv_status = pickle.loads(data)
            logger.info(f"received upkv_status {upkv_status} from {(client_ip, client_port)}")
            await g_objs.httpserver_manager.update_req_status(upkv_status)
    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"kv_move_status client {(client_ip, client_port)} has error {str(e)}")
        logger.exception(str(e))
    return
