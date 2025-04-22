from typing import List
import zmq

import torch.multiprocessing as mp

from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip

from .pd_remote_prefill_obj import (
    ConnectRequest,
    RemoteRequest,
    RemoteRequstType,
    PrefillRequest,
    RemotePrefillRequest,
    RemotePrefillServerInfo,
    RemotePrefillTask,
    RemotePrefillStatus
)
from .nixl_kv_transporter import NixlMetadata


logger = init_logger(__name__)

class PDRemotePrefillBase:
    def __init__(self,
                 id: int,
                 from_backend_queue: mp.Queue,
                 to_backend_queues: List[mp.Queue],
                 agent_meta_queues: List[mp.Queue], # need send kv cache to this process and register with nixl
                 ):
        self.id = id
        self.tp_size = len(agent_meta_queues)
        self.agent_meta_queues = agent_meta_queues
        self.from_backend_queue = from_backend_queue
        self.to_backend_queues = to_backend_queues
        self.local_agent_meta  = None

    def local_init(self):
        agent_metas = NixlMetadata(id=self.id, agent_metadatas=[], num_tokens=[], agent_mem_descs=[])
        for tp in range(self.tp_size):
            agent_metadata, num_tokens, mem_desc = self.agent_meta_queues[tp].get(timeout=60)
            logger.info(f"Received agent_metadata from {tp} with mem reg: {mem_desc}")
            agent_metas.num_tokens.append(num_tokens)
            agent_metas.agent_metadatas.append(agent_metadata)
            agent_metas.agent_mem_descs.append(mem_desc)

        self.local_agent_meta = agent_metas
        logger.info("All local kv cache registered.")


class PDRemotePrefillServer(PDRemotePrefillBase):
    def __init__(self,
                 id: int,
                 http_server_port: int,
                 server_port: int,
                 from_backend_queue: mp.Queue,
                 to_backend_queues: List[mp.Queue],
                 agent_meta_queues: List[mp.Queue]):
        super().__init__(id, from_backend_queue, to_backend_queues, agent_meta_queues)
        # map from client id to decode server info
        self.remote_decode_clients = {}

        # build control path
        _ctx = zmq.Context()
        self.recv_from_decode = _ctx.socket(zmq.PAIR)
        self.host_ip = get_hostname_ip()
        self.recv_from_decode.bind(f"tcp://{self.host_ip}:{server_port}")

        # build trigger remote prefill path
        self.send_to_httpserver = _ctx.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://{self.host_ip}:{http_server_port}")

        self.prefill_requests = {}



    def main_loop(self):
        self.local_init()
        while True:
            request: RemoteRequest = self.recv_from_decode.recv_pyobj()
            logger.info(f"recevied request from decode, type: {request.type}")

            # forward request to prefill server
            for queue in self.to_backend_queues:
                queue.put(request)

            if request.type == RemoteRequstType.REMOTE_CONNECT:
                success = True
                for idx in range(self.tp_size):
                    ack = self.from_backend_queue.get()
                    if ack != "OK":
                        success = False
                        break

                self.recv_from_decode.send_pyobj(success)
                if not success:
                    logger.warning(f"Remote connect failed: {request}")


            if request.type == RemoteRequstType.REMOTE_PREFILL:
                request: PrefillRequest = request
                self.trigger_prefill(request)



    def trigger_prefill(self, request: PrefillRequest):
        self.send_to_httpserver.send_pyobj((request.data.prompt, request.data.sampling_params, request.data.multimodal_params))
        self.prefill_requests[request.data.sampling_params.group_request_id] = request



class PDRemotePrefillClient(PDRemotePrefillBase):

    def __init__(self,
                 id: int,
                 from_backend_queue: mp.Queue, # only tp0 will trigger prefill
                 to_backend_queues: List[mp.Queue], # one to many done queue
                 agent_meta_queues: List[mp.Queue]
                 ):
        super().__init__(id, from_backend_queue, to_backend_queues, agent_meta_queues)
        # map from server id to prefill server info

        self.remote_prefill_servers = {}

    def connect_to_prefill_server(self, server_info: RemotePrefillServerInfo):
        # build control path if not exist
        if server_info.perfill_server_id not in self.remote_prefill_servers:
            _ctx = zmq.Context()
            _socket = _ctx.socket(zmq.PAIR)
            connect_str = f"tcp://{server_info.prefill_server_ip}:{server_info.prefill_server_port}"
            _socket.connect(connect_str)
            _socket.send_pyobj(ConnectRequest(
                type=RemoteRequstType.REMOTE_CONNECT,
                decode_id=self.id,
                num_tokens=self.local_agent_meta.num_tokens,
                agent_metadatas=self.local_agent_meta.agent_metadatas,
                agent_mem_descs=self.local_agent_meta.agent_mem_descs))

            success = _socket.recv_pyobj()
            if success:
                self.remote_prefill_servers[server_info.perfill_server_id] = (_socket, server_info)
                return True
            else:
                logger.warning("Remote Prefill Server Connect Failed")
                return False

        return True

    def main_loop(self):
        self.local_init()
        while True:
            prefill_tasks: RemotePrefillTask = self.from_backend_queue.get()

            # connect first
            if(self.connect_to_prefill_server(prefill_tasks.server_info)):
                # do prefill
                self.remote_prefill(prefill_tasks.server_info.perfill_server_id, prefill_tasks.prefill_request)
            else:
                # failed to connect a remote
                for idx in self.to_backend_queues:
                    self.to_backend_queues.put(RemotePrefillStatus(
                        group_req_id=prefill_tasks.prefill_request.sampling_params.group_request_id,
                        status=-1,
                    ))

    # place request to server do remote prefill
    def remote_prefill(self, server_id: int, prefill_request: RemotePrefillRequest):
        socket, _ = self.remote_prefill_servers[server_id]
        prefill_request.sampling_params.max_new_tokens = 1
        socket.send_pyobj(PrefillRequest(type=RemoteRequstType.REMOTE_PREFILL, decode_id=self.id, data=prefill_request))



def remote_prefill_server_loop(
    id: int,
    http_server_port: int,
    server_port: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):
    server = PDRemotePrefillServer(id, http_server_port, server_port,
                                   from_backend_queue, to_backend_queues, agent_meta_queues)
    server.main_loop()


def start_pd_remote_prefill_server_process(
    id: int,
    http_server_port: int,
    server_port: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):
    proc = mp.Process(
        target=remote_prefill_server_loop,
        args=(
            id, http_server_port, server_port,
            from_backend_queue, to_backend_queues, agent_meta_queues)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"remote prefill server with id: {id} started!")
    return proc


def remote_prefill_client_loop(
    id: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue]):

    client = PDRemotePrefillClient(
            id,
            from_backend_queue,
            to_backend_queues,
            agent_meta_queues,
        )
    client.main_loop()

def start_pd_remote_prefill_client_process(
    id: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue]
):

    proc = mp.Process(
        target=remote_prefill_client_loop,
        args=(id, from_backend_queue, to_backend_queues, agent_meta_queues)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"remote prefill client with id: {id} started!")
    return proc
