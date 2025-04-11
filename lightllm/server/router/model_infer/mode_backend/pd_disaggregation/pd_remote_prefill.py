import asyncio
from typing import List, Union
from enum import Enum
import zmq
from collections import defaultdict

from torch import Tensor
import torch.multiprocessing as mp

from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.net_utils import get_hostname_ip

from .pd_remote_prefill_obj import (
    ConnectRequest,
    RemoteRequest,
    RemoteAgent,
    RemoteRequstType,
    PrefillRequest,
    RemotePrefillRequest,
    RemotePrefillServerInfo,
    KVMoveRequest,
    RemotePrefillTask,
)


logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent

except ImportError:
    logger.error("nixl is not installed, which is required for pd disagreggation!!!")
    raise


class PDRemotePrefillBase:
    def __init__(self,
                 device_index: int,
                 kv_cache_queue: mp.Queue, # need send kv cache to this process and register with nixl
                 tp_size: int,):
        # create local nixl agent
        self.nixl_agent_name = f'nixl_agent_{get_unique_server_name}_{device_index}'
        self.nixl_agent = nixl_agent(self.nixl_agent_name, None)

        # metadata need to send to remote server to make connection
        self.nixl_agent_metadata = self.nixl_agent.get_agent_metadata()

        self.reg_descs = [None] * tp_size
        self.local_xfer_handles = [None] * tp_size

        self.device_index = device_index
        self.kv_cache_queue = kv_cache_queue
        self.tp_size = tp_size

        self.num_layers = -1
        self.num_tokens = -1
        self.num_heads = -1
        self.head_dims= -1
        self.token_len = -1
        self.layer_len = -1


    def _create_xfer_handles(self, idx, reg_descs):
        base_addr, _, device_id = reg_descs[0]
        tokens_data = []
        for layer_id in range(self.num_layers):
            for token_id in range(self.num_tokens):
                tokens_data.append(base_addr + layer_id * self.layer_len + token_id * self.token_len, self.token_len, device_id)

        descs = self.nixl_agent.get_xfer_descs(tokens_data, "VRAM", True)
        self.local_xfer_handles[idx] = self.nixl_agent.prep_xfer_dlist("", descs, is_sorted=True)

    def _get_token_desc_ids(self, token_ids: List[int]):
        descs_ids = []
        for layer_id in range(self.num_layers):
            for token_id in token_ids:
                descs_ids.append(layer_id * self.num_tokens + token_id)
        return descs_ids

    def local_init(self):
        for _ in range(self.tp_size):
            idx, tensor = self.kv_cache_queue.get(timeout=60)
            if self.num_layers == -1:
                self.num_layers, self.num_tokens, self.num_heads, self.head_dim = tensor.shape
                self.token_len = self.num_heads * self.head_dim * tensor.element_size()
                self.layer_len = self.num_tokens * self.token_len

            self.reg_descs[idx] = self.nixl_agent.register_memory(tensor)
            self._create_xfer_handles(idx, self.reg_descs[idx])

        logger.info("All local kv cache registered.")


class PDRemotePrefillServer(PDRemotePrefillBase):
    def __init__(self,
                 http_server_port: int,
                 server_port: int,
                 device_index: int,
                 kvmove_request_queue: mp.Queue,
                 kvmove_done_queue: mp.Queue,
                 kv_cache_queue: mp.Queue,
                 tp_size: int):
        super().__init__(device_index, kv_cache_queue, tp_size)
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

        self.kvmove_request_queue = kvmove_request_queue
        self.kvmove_done_queue = kvmove_done_queue

        self.inflight_transfer = defaultdict(list)

    def add_remote_agent(self, request: ConnectRequest):
        peer_name = self.nixl_agent.add_remote_agent(request.agent_metadata)
        mem_desc = self.nixl_agent.deserialize_descs(request.agent_mem_desc)
        kv_xfer_handles = []
        for idx, desc in enumerate(mem_desc):
            kv_xfer_handles.append(self._create_xfer_handles(idx, desc))

        self.remote_decode_clients[request.decode_id] = RemoteAgent(
            name=peer_name,
            kv_mem_desc=mem_desc,
            kv_xfer_handles=kv_xfer_handles)


    def main_loop(self):
        self.local_init()
        self.transfer_task = asyncio.create_task(self.transfer_loop())
        self.wait_transfer_task = asyncio.create_task(self.wait_transfer_task())
        while True:
            request: RemoteRequest = self.recv_from_decode.recv_pyobj()
            if request.type == RemoteRequstType.REMOTE_CONNECT:
                request: ConnectRequest = request
                self.add_remote_agent(request)
            elif request.type == RemoteRequstType.REMOTE_PREFILL:
                request: PrefillRequest = request
                self.trigger_prefill(request)


    def trigger_prefill(self, request: PrefillRequest):
        self.send_to_httpserver.send_pyobj((request.data.prompt, request.data.sampling_params, request.data.multimodal_params))
        self.prefill_requests[request.data.sampling_params.group_request_id] = request


    async def transfer_loop(self):
        while True:
            request: KVMoveRequest = self.kv_cache_queue.get()
            await self.trigger_kvcache_write(request)


    async def trigger_kvcache_write(self, request: KVMoveRequest):
        group_reqeust_id = request.group_req_id
        prefill_request: PrefillRequest = self.prefill_requests[group_reqeust_id]
        skip_kv_move_len = prefill_request.data.local_cached_len
        src_token_ids = request.token_ids[skip_kv_move_len:]
        dst_token_ids = prefill_request.data.token_ids[skip_kv_move_len:]
        remote_agent = self.remote_decode_clients[prefill_request.decode_id]
        if len(src_token_ids) > 0:
            assert len(src_token_ids) == len(dst_token_ids)
            src_token_descs = self._get_token_desc_ids(src_token_ids)
            dst_token_descs = self._get_token_desc_ids(dst_token_ids)

            for i in range(self.tp_size): #TODO make this a single transfer
                src_handle = self.local_xfer_handles[i]
                dst_handle = remote_agent.remote_xfer_handles[i]
                handle = self.nixl_agent.make_prepped_xfer("WRITE",
                                                           src_handle, src_token_descs,
                                                           dst_handle, dst_token_descs, group_reqeust_id)
                self.inflight_transfer[group_reqeust_id].append(handle)
                status = self.nixl_agent.transfer(handle)


            await self.kv_cache_queue.put({"src": src_token_descs, "dst": dst_token_descs})


    def get_done_tranfers(self) -> List[str]:
        done_req_ids = []
        failed_req_ids = []
        for req_id, handles in self.inflight_transfer.items():
            running_reqs = []
            failed_reqs = []
            for handle in handles:
                xfer_state = self.nixl_agent.check_xfer_state(handle)
                if xfer_state == "DONE":
                    self.nixl_wrapper.release_xfer_handle(handle) # TODO ptarasiewicz: why abort is throwing errors?
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    logger.warning(f"Transfer failed with state {xfer_state}")
                    failed_reqs.append(handle)
                    break

            if failed_reqs:
                failed_req_ids.append(req_id)
                continue

            if len(running_reqs) == 0:
                done_req_ids.append(req_id)
            else:
                self.inflight_transfer[req_id] = running_reqs

        return done_req_ids, failed_req_ids


    async def wait_transfer_loop(self):
        while True:
            done_ids, failed_ids = self.get_done_transfers()
            # handle successfully completed transfers
            pass

            # handle failed transfers
            pass

            # remote ids from inflight transfers and cancle inflight transfers if failed






class PDRemotePrefillClient(PDRemotePrefillBase):

    def __init__(self,
                 prefill_request_queue: mp.Queue, # only tp0 will trigger prefill
                 prefill_done_queue: List[mp.Queue], # one to many done queue
                 device_index: int,
                 kv_cache_queue: mp.Queue, # need send kv cache to this process and register with nixl
                 tp_size: int,
                 my_id: int,
                 ):
        super().__init__(device_index, kv_cache_queue, tp_size)
        # map from server id to prefill server info
        self.remote_prefill_servers = {}
        self.prefill_request_queue = prefill_request_queue
        self.prefill_done_queue = prefill_done_queue
        self.remote_prefill_requests = {}
        self.my_id = my_id

    def connect_to_prefill_server(self, server_info: RemotePrefillServerInfo):
        # build control path if not exist
        if server_info.perfill_server_id not in self.remote_prefill_servers:
            _ctx = zmq.Context()
            _socket = _ctx.socket(zmq.PUSH)
            connect_str = f"tcp://{server_info.prefill_server_ip}:{server_info.prefill_server_port}"
            _socket.connect(connect_str)
            _socket.send_pyobj(ConnectRequest(
                type=RemoteRequstType.REMOTE_CONNECT,
                decode_id=self.my_id,
                agent_metadata=self.nixl_agent_metadata,
                agent_mem_desc=self.nixl_agent.get_serialized_descs(self.reg_descs)))
            self.remote_prefill_servers[server_info.perfill_server_id] = (_socket, server_info)

    def main_loop(self):
        self.local_init()
        asyncio.create_task(self.prefill_wait_loop())
        while True:
            prefill_tasks: List[RemotePrefillTask] = self.prefill_request_queue.get()
            for task in prefill_tasks:
                # connect first
                self.connect_to_prefill_server(task.server_info)
                # do prefill
                self.remote_prefill(task.server_info.perfill_server_id, task.prefill_request)


    # place request to server do remote prefill
    def remote_prefill(self, server_id: int, prefill_request: RemotePrefillRequest):
        socket, _ = self.remote_prefill_servers[server_id]
        group_req_id = str(prefill_request.sampling_params.group_request_id)
        socket.send_pyobj(RemoteRequest(type=RemoteRequstType.REMOTE_PREFILL, decode_id=self.my_id, data=prefill_request))
        self.remote_prefill_requests[group_req_id] = prefill_request


    async def prefill_wait_loop(self):
        while True:
            notifies = self.nixl_agent.get_new_notifs()
            for agent_name, msgs in notifies.items():
                for msg in msgs:
                    # we got a finished prefill msg
                    for pdq in self.prefill_done_queue:
                        pdq.put(msg)
                    del self.remote_prefill_requests[msg]
                    logger.info(f"prefill reqeust: {msg} done")
