import time
import asyncio
import base64
import pickle
import setproctitle
import multiprocessing as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from threading import Lock
from typing import Dict, List
from fastapi.responses import JSONResponse
from lightllm.utils.log_utils import init_logger
from ..pd_io_struct import PD_Master_Obj
from .nccl_tcp_store import start_tcp_store_server
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.process_check import start_parent_check_thread


logger = init_logger(__name__)
app = FastAPI()

registered_pd_master_objs: Dict[str, PD_Master_Obj] = {}
registered_pd_master_obj_lock = Lock()

global_req_id = 0
global_req_id_lock = Lock()

# This is a global ID for multimodal embedding, starting from 100000000
global_multimodal_embedding_id = 100000000
global_multimodal_embedding_id_lock = Lock()


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    return JSONResponse({"message": "Ok"}, status_code=200)


@app.websocket("/pd_master_register")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"ws connected from IP: {client_ip}, Port: {client_port}")
    registered_pd_master_obj: PD_Master_Obj = pickle.loads(await websocket.receive_bytes())
    logger.info(f"recieved registered_pd_master_obj {registered_pd_master_obj}")
    with registered_pd_master_obj_lock:
        registered_pd_master_objs[registered_pd_master_obj.node_id] = registered_pd_master_obj

    try:
        while True:
            data = await websocket.receive_text()
            assert data == "heartbeat"
    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"registered_pd_master_obj {registered_pd_master_obj} has error {str(e)}")
        logger.exception(str(e))
    finally:
        logger.error(f"registered_pd_master_obj {registered_pd_master_obj} removed")
        with registered_pd_master_obj_lock:
            registered_pd_master_objs.pop(registered_pd_master_obj.node_id, None)
    return


@app.get("/registered_objects")
async def get_registered_objects():
    with registered_pd_master_obj_lock:
        serialized_data = pickle.dumps(registered_pd_master_objs)
        base64_encoded = base64.b64encode(serialized_data).decode("utf-8")
        return {"data": base64_encoded}


@app.get("/allocate_global_unique_id_range")
async def allocate_global_id_range():
    """
    Allocate a global ID range for the requesting client without requiring parameters.

    Returns:
        dict: A dictionary containing the start and end of the allocated ID range.

    Example HTTP client usage:
    ```python
    response = requests.get("http://<server_address>/allocate_global_unique_id_range")
    print(response.json())
    ```
    """
    global global_req_id
    range_size = 800000
    with global_req_id_lock:
        if global_req_id + range_size > 2 ** 63 - 1:
            global_req_id = 0
        start_id = global_req_id
        global_req_id += range_size
        end_id = global_req_id

    return {"start_id": start_id, "end_id": end_id}


@app.get("/allocate_global_unique_multimodal_id_range")
async def allocate_global_unique_multimodal_id_range():
    global global_multimodal_embedding_id
    range_size = 8000000
    with global_multimodal_embedding_id_lock:
        if global_multimodal_embedding_id + range_size > 2 ** 63 - 1:
            global_multimodal_embedding_id = 100000000
        start_id = global_multimodal_embedding_id
        global_multimodal_embedding_id += range_size
        end_id = global_multimodal_embedding_id

    return {"start_id": start_id, "end_id": end_id}


global_store_port_to_process: Dict[int, mp.Process] = {}
global_store_port_to_client_states: Dict[int, List[bool]] = {}
global_store_port_lock = asyncio.Lock()


@app.get("/start_tcp_store_server")
async def http_start_tcp_store_server(
    tcp_store_port: int = Query(...), rank_id: int = Query(...), world_size: int = Query(...)
):
    """
    Start a TCP store server for NCCL communication.

    Args:
        tcp_store_port (int): The port number for the TCP store server.
        rank_id (int): The rank ID of inference process.
        world_size (int): The world size of nccl group.

    Returns:
        dict: A dictionary containing the status of the server.
    """
    global global_store_port_to_process
    global global_store_port_to_client_states
    global global_store_port_lock

    args = get_env_start_args()

    if rank_id == 0:
        async with global_store_port_lock:
            if tcp_store_port in global_store_port_to_client_states:
                logger.error(f"tcp store server {tcp_store_port} already started, rank_id 0 find client state exists")
                assert False, f"tcp store server {tcp_store_port} already started, rank_id 0 find client state exists"

            if tcp_store_port in global_store_port_to_process:
                logger.warning(f"tcp store server {tcp_store_port} already started, kill and restart it")
                process = global_store_port_to_process[tcp_store_port]
                process.kill()
                process.join()

            global_store_port_to_process[tcp_store_port] = start_tcp_store_server(
                args.config_server_host, tcp_store_port
            )

            world_size_state = [True for _ in range(world_size)]
            global_store_port_to_client_states[tcp_store_port] = world_size_state

        world_size_state[rank_id] = False

        start_time = time.time()
        while any(world_size_state):
            await asyncio.sleep(1)
            if time.time() - start_time > 60 * 3:
                logger.error(
                    f"tcp store server {tcp_store_port} rank_id {rank_id} world_size {world_size} wait all quit timeout"
                )
                async with global_store_port_lock:
                    global_store_port_to_client_states.pop(tcp_store_port, None)
                raise Exception(
                    f"tcp store server {tcp_store_port} rank_id {rank_id} world_size {world_size} wait timeout"
                )

        async with global_store_port_lock:
            global_store_port_to_client_states.pop(tcp_store_port, None)

        return {"status": "ok"}
    else:
        start_time = time.time()
        while tcp_store_port not in global_store_port_to_client_states:
            await asyncio.sleep(1)
            if time.time() - start_time > 60 * 3:
                logger.error(f"tcp store port {tcp_store_port} rank_id {rank_id} world_size {world_size} state timeout")
                raise Exception(
                    f"tcp store server {tcp_store_port} rank_id {rank_id} world_size {world_size} state timeout"
                )

        world_size_state = global_store_port_to_client_states[tcp_store_port]

        assert (
            world_size_state[rank_id] is True
        ), f"tcp store server {tcp_store_port} rank_id {rank_id} world_size {world_size} world_size_state error"
        world_size_state[rank_id] = False
        return {"status": "ok"}


setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::config_server")
logger.info("config server start_parent_check_thread...")
start_parent_check_thread()
