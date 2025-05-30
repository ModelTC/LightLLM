from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import List, Union, Optional, Any
from threading import Lock
import pickle
import zmq

from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import RemotePrefillServerInfo

logger = init_logger(__name__)

try:
    from nixl._api import nixlBind, nixl_prepped_dlist_handle, nixl_xfer_handle

except ImportError:
    logger.error("nixl is not installed, which is required for pd disagreggation!!!")
    raise


class RemoteRequstType(Enum):
    REMOTE_CONNECT = 1
    REMOTE_PREFILL = 2


@dataclass
class RemotePrefillRequest:
    prompt: Union[str, List[int]]
    sampling_params: SamplingParams
    multimodal_params: MultimodalParams
    local_cached_len: int  # will skip transfer
    token_ids: List[int]  # mem cache indexes


@dataclass
class RemotePrefillTask:
    server_info: RemotePrefillServerInfo
    prefill_request: RemotePrefillRequest


@dataclass
class RemoteRequest:
    type: RemoteRequstType


@dataclass
class ConnectRequest(RemoteRequest):
    decode_id: int
    num_tokens: List[int]
    agent_metadatas: List[bytes]
    agent_mem_descs: List[bytes]


@dataclass
class TransferState:
    start_time: float
    current_kv_len: int
    current_chunk_id: int


@dataclass
class PrefillRequest(RemoteRequest):
    decode_id: int
    data: RemotePrefillRequest
    # transfer status
    transfer_state: Optional[TransferState]


@dataclass
class KVMoveRequest:
    group_req_id: int
    token_ids: List[int]
    prev_kv_len: int
    cur_kv_len: int


@dataclass
class RemoteAgent:
    name: str
    num_tokens: int
    kv_mem_desc: nixlBind.nixlRegDList
    kv_xfer_handles: nixl_prepped_dlist_handle


@dataclass
class KVMoveRequestState:
    handles: List[nixl_xfer_handle]
    done_handles: List[nixl_xfer_handle]
    remote_agent: RemoteAgent
    abort: bool


@dataclass
class RemotePrefillStatus:
    group_req_id: int
    status: int
    chunk_id: int
    is_last: bool

    def serialize(self):
        return json.dumps(asdict(self)).encode()

    @classmethod
    def deserialize(cls, data: bytes):
        return cls(**json.loads(data.decode()))


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def pop(self, key: Any, default: Optional[Any] = None) -> Any:
        with self._lock:
            return self._dict.pop(key, default)

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def clear(self) -> None:
        with self._lock:
            self._dict.clear()


class SockWithPoller:
    def __init__(self, sock: zmq.Socket):
        self.sock = sock
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv_pyobj(self, timeout: int = 5):
        socks = dict(self.poller.poll(timeout * 1000))
        if socks:
            if socks.get(self.sock) == zmq.POLLIN:
                return self.sock.recv_pyobj()
        else:
            None

    def send_pyobj(self, obj: Any):
        return self.sock.send_pyobj(obj)

    def recv_pyobj_multipart(self):
        client_id, data = self.sock.recv_multipart()
        return client_id, pickle.loads(data)

    def send_pyobj_multipart(self, client_id: bytes, data: Any):
        return self.sock.send_multipart([client_id, pickle.dumps(data)])

    def bind(self, addr: str):
        return self.sock.bind(addr)

    def connect(self, addr: str):
        return self.sock.connect(addr)
