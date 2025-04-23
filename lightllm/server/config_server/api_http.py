from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from threading import Lock
from typing import Dict
from fastapi.responses import JSONResponse
from lightllm.utils.log_utils import init_logger
from ..pd_io_struct import PD_Master_Obj
import base64
import pickle
import os
import requests

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
