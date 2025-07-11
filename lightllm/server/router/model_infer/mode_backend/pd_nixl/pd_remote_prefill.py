from typing import List, Any
import zmq
import inspect
import random
import time

import torch.multiprocessing as mp

from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_hostname_ip
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.pd_io_struct import DistInfo

from .pd_remote_prefill_obj import (
    ConnectRequest,
    RemoteRequest,
    RemoteRequstType,
    PrefillRequest,
    RemotePrefillRequest,
    RemotePrefillServerInfo,
    RemotePrefillTask,
    RemotePrefillStatus,
    RemoteTransferStatusType,
    RemoteTransferType,
    SockWithPoller,
)
from .nixl_kv_transporter import NixlMetadata

logger = init_logger(__name__)


class PDRemotePrefillBase:
    def __init__(
        self,
        id: int,
        dist_info: DistInfo,
        from_backend_queue: mp.Queue,
        to_backend_queues: List[mp.Queue],
        agent_meta_queues: List[mp.Queue],  # need send kv cache to this process and register with nixl
    ):
        self.id = id
        self.dist_info = dist_info
        assert len(agent_meta_queues) == dist_info.node_world_size
        self.agent_meta_queues = agent_meta_queues
        self.from_backend_queue = from_backend_queue
        self.to_backend_queues = to_backend_queues
        self.local_agent_meta = None

    def local_init(self):
        agent_metas = NixlMetadata(
            id=self.id,
            agent_metadatas=[],
            num_tokens=[],
            num_pages=[],
            agent_mem_descs=[],
            agent_page_mem_descs=[],
        )
        for tp in range(self.dist_info.node_world_size):
            agent_metadata, num_tokens, num_pages, mem_desc, page_mem_desc = self.agent_meta_queues[tp].get(timeout=60)
            logger.info(f"Received agent_metadata from {tp} with mem reg: {mem_desc}")
            agent_metas.num_tokens.append(num_tokens)
            agent_metas.num_pages.append(num_pages)
            agent_metas.agent_metadatas.append(agent_metadata)
            agent_metas.agent_mem_descs.append(mem_desc)
            agent_metas.agent_page_mem_descs.append(page_mem_desc)

        self.local_agent_meta = agent_metas
        logger.info("All local kv cache registered.")


class PDRemotePrefillServer(PDRemotePrefillBase):
    def __init__(
        self,
        id: int,
        dist_info: DistInfo,
        http_server_port: int,
        server_port: int,
        from_backend_queue: mp.Queue,
        to_backend_queues: List[mp.Queue],
        agent_meta_queues: List[mp.Queue],
    ):
        super().__init__(id, dist_info, from_backend_queue, to_backend_queues, agent_meta_queues)
        # map from client id to decode server info
        self.remote_decode_clients = {}

        # build control path
        _ctx = zmq.Context()
        self.recv_from_decode = SockWithPoller(_ctx.socket(zmq.ROUTER))
        self.host_ip = get_hostname_ip()
        self.recv_from_decode.bind(f"tcp://{self.host_ip}:{server_port}")

        # build trigger remote prefill path
        self.send_to_httpserver = SockWithPoller(_ctx.socket(zmq.PUSH))
        self.send_to_httpserver.connect(f"tcp://{self.host_ip}:{http_server_port}")

    def main_loop(self):
        self.local_init()
        while True:
            try:
                client_obj, request = self.recv_from_decode.recv_pyobj_multipart()
                request: RemoteRequest
                logger.info(f"recevied request from decode, type: {request.type}")

                if request.type == RemoteRequstType.REMOTE_CONNECT:
                    # forward request to all prefill server
                    for queue in self.to_backend_queues:
                        queue.put(request)

                    success = True
                    for idx in range(self.dist_info.node_world_size):
                        ack = self.from_backend_queue.get()
                        logger.info(f"received ack from backend {idx}: {ack}")
                        if ack != "OK":
                            success = False
                            break

                    self.recv_from_decode.send_pyobj_multipart(client_obj, success)
                    logger.info(f"Sent ack to decode: {success}")
                    if not success:
                        logger.warning(f"Remote connect failed: {request}")

                if request.type == RemoteRequstType.REMOTE_PREFILL:
                    request: PrefillRequest = request
                    if self.dist_info.dp_size_in_node > 1:
                        group_req_id = request.data.sampling_params.group_request_id
                        suggested_dp_index = request.data.sampling_params.suggested_dp_index
                        if suggested_dp_index < 0:  # not likely to happen
                            suggested_dp_index = random.randint(0, self.dist_info.dp_size_in_node)
                            request.data.sampling_params.suggested_dp_index = suggested_dp_index
                            logger.warning(
                                f"Suggested dp index is negative for {group_req_id}, set to {suggested_dp_index}"
                            )

                        for local_rank in range(
                            suggested_dp_index * self.dist_info.dp_world_size,
                            (suggested_dp_index + 1) * self.dist_info.dp_world_size,
                        ):
                            self.to_backend_queues[local_rank].put(request)
                    else:
                        for queue in self.to_backend_queues:
                            queue.put(request)

                    self.send_to_httpserver.send_pyobj(
                        (request.data.prompt, request.data.sampling_params, request.data.multimodal_params)
                    )

            except Exception as e:
                logger.error(f"Error in remote prefill server loop: {e}", exc_info=e)


class PDRemotePrefillClient(PDRemotePrefillBase):
    def __init__(
        self,
        id: int,
        dist_info: DistInfo,
        from_backend_queue: mp.Queue,  # only tp0 will trigger prefill
        to_backend_queues: List[mp.Queue],  # one to many done queue
        agent_meta_queues: List[mp.Queue],
    ):
        super().__init__(id, dist_info, from_backend_queue, to_backend_queues, agent_meta_queues)
        # map from server id to prefill server info

        self.remote_prefill_servers = {}
        self.client_socket_cnt = 0
        self._ctx = zmq.Context()

    def _connect_server(self, server_ip: str, port: int):
        _socket = self._ctx.socket(zmq.DEALER)
        _socket.setsockopt_string(zmq.IDENTITY, f"{self.id}_{self.client_socket_cnt}")
        self.client_socket_cnt += 1
        connect_str = f"tcp://{server_ip}:{port}"
        _socket.connect(connect_str)
        return SockWithPoller(_socket)

    def _send_nixl_agent(self, socket: SockWithPoller):
        socket.send_pyobj(
            ConnectRequest(
                type=RemoteRequstType.REMOTE_CONNECT,
                decode_id=self.id,
                num_tokens=self.local_agent_meta.num_tokens,
                num_pages=self.local_agent_meta.num_pages,
                agent_metadatas=self.local_agent_meta.agent_metadatas,
                agent_mem_descs=self.local_agent_meta.agent_mem_descs,
                agent_page_mem_descs=self.local_agent_meta.agent_page_mem_descs,
            )
        )

        success = socket.recv_pyobj(timeout=60)
        logger.info(f"recv remote nixl connect response {success}")
        if success is None:
            logger.warning("timeout to recv remote nixl connect response")
            return False

        return success

    def connect_to_prefill_server(self, server_info: RemotePrefillServerInfo):

        if server_info.perfill_server_id in self.remote_prefill_servers:
            return True

        # build control path if not exist
        _socket = self._connect_server(server_info.prefill_server_ip, server_info.prefill_server_port)
        success = self._send_nixl_agent(_socket)
        if success:
            self.remote_prefill_servers[server_info.perfill_server_id] = (_socket, server_info)
            return True
        else:
            logger.warning("Remote Prefill Server Connect Failed")
            return False

    def main_loop(self):
        self.local_init()
        while True:
            try:
                prefill_tasks: RemotePrefillTask = self.from_backend_queue.get()
                # connect first
                if self.connect_to_prefill_server(prefill_tasks.server_info):
                    # do prefill
                    self.remote_prefill(prefill_tasks.server_info.perfill_server_id, prefill_tasks.prefill_request)
                else:
                    # failed to connect a remote
                    for idx in self.to_backend_queues:
                        self.to_backend_queues.put(
                            RemotePrefillStatus(
                                transfer_type=RemoteTransferType.PAGE_TRANSFER,
                                group_req_id=prefill_tasks.prefill_request.sampling_params.group_request_id,
                                status=RemoteTransferStatusType.FAILED,
                                is_last=True,
                            )
                        )
            except Exception as e:
                logger.error(f"Remote prefill client loop error: {e}", exc_info=e)

    # place request to server do remote prefill
    def remote_prefill(self, server_id: int, prefill_request: RemotePrefillRequest):
        socket, _ = self.remote_prefill_servers[server_id]
        prefill_request.sampling_params.max_new_tokens = 1
        socket.send_pyobj(
            PrefillRequest(
                type=RemoteRequstType.REMOTE_PREFILL, decode_id=self.id, data=prefill_request, transfer_state=None
            )
        )


def remote_prefill_server_loop(
    id: int,
    dist_info: DistInfo,
    http_server_port: int,
    server_port: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):
    graceful_registry(inspect.currentframe().f_code.co_name)
    server = PDRemotePrefillServer(
        id, dist_info, http_server_port, server_port, from_backend_queue, to_backend_queues, agent_meta_queues
    )
    server.main_loop()


def start_pd_remote_prefill_server_process(
    id: int,
    dist_info: DistInfo,
    http_server_port: int,
    server_port: int,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):
    proc = mp.Process(
        target=remote_prefill_server_loop,
        args=(id, dist_info, http_server_port, server_port, from_backend_queue, to_backend_queues, agent_meta_queues),
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"remote prefill server with id: {id} started!")
    return proc


def remote_prefill_client_loop(
    id: int,
    dist_info: DistInfo,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):
    graceful_registry(inspect.currentframe().f_code.co_name)

    client = PDRemotePrefillClient(
        id,
        dist_info,
        from_backend_queue,
        to_backend_queues,
        agent_meta_queues,
    )
    client.main_loop()


def start_pd_remote_prefill_client_process(
    id: int,
    dist_info: DistInfo,
    from_backend_queue: mp.Queue,
    to_backend_queues: List[mp.Queue],
    agent_meta_queues: List[mp.Queue],
):

    proc = mp.Process(
        target=remote_prefill_client_loop,
        args=(id, dist_info, from_backend_queue, to_backend_queues, agent_meta_queues),
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"remote prefill client with id: {id} started!")
    return proc
