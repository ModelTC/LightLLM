from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import List, Union, Optional, Any
from threading import Lock, Condition
import pickle
import zmq
import threading

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
    page_ids: List[int]  # transfer page indexes


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
    num_pages: List[int]
    agent_metadatas: List[bytes]
    agent_mem_descs: List[bytes]
    agent_page_mem_descs: List[bytes]


@dataclass
class TransferState:
    start_time: float
    lock: threading.Lock
    free_page_ids: List[int]

    current_kv_len: int = 0
    current_chunk_id: int = 0

    transfered_kv_len: int = 0
    transfered_chunk_id: int = 0

    token_index: List[int] = None
    is_finished: bool = False

    next_token_id: int = None
    next_token_logprob: float = None

    def completed(self):
        return self.is_finished and self.transfered_kv_len == self.current_kv_len


@dataclass
class PrefillRequest(RemoteRequest):
    decode_id: int
    data: RemotePrefillRequest
    # transfer status
    transfer_state: Optional[TransferState]


@dataclass
class KVMoveRequest:
    group_req_id: int
    prev_kv_len: int
    cur_kv_len: int


@dataclass
class RemoteAgent:
    name: str
    num_tokens: int
    num_pages: int
    kv_mem_desc: nixlBind.nixlRegDList
    kv_xfer_handles: nixl_prepped_dlist_handle
    kv_page_mem_desc: nixlBind.nixlRegDList
    kv_page_xfer_handles: nixl_prepped_dlist_handle


@dataclass
class KVMoveRequestState:
    handles: List[nixl_xfer_handle]
    done_handles: List[nixl_xfer_handle]
    remote_agent: RemoteAgent
    abort: bool
    is_last_arrived: bool


class SerializableBase:
    def to_dict(self):
        return asdict(self)

    def serialize(self):
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_dict(cls, dict_obj):
        return cls(**dict_obj)

    @classmethod
    def deserialize(cls, data: bytes):
        return cls.from_dict(json.loads(data.decode()))


class RemoteTransferType(Enum):
    TOKEN_TRANSFER = 1
    PAGE_TRANSFER = 2


class RemoteTransferStatusType(Enum):
    FAILED = -1
    IN_PROGRESS = 0
    SUCCESS = 1


@dataclass
class RemotePrefillStatus(SerializableBase):
    transfer_type: RemoteTransferType
    group_req_id: int
    status: RemoteTransferStatusType
    chunk_id: int = -1
    is_last: bool = False
    page_id: int = -1
    kv_start: int = 0
    kv_len: int = 0
    next_token_id: int = None
    next_token_logprob: float = None

    def to_dict(self):
        dict_obj = asdict(self)
        dict_obj["transfer_type"] = self.transfer_type.name
        dict_obj["status"] = self.status.name
        return dict_obj

    @classmethod
    def from_dict(cls, dict_obj):
        dict_obj["transfer_type"] = RemoteTransferType[dict_obj["transfer_type"]]
        dict_obj["status"] = RemoteTransferStatusType[dict_obj["status"]]
        return cls(**dict_obj)


@dataclass
class PageTransferAck(SerializableBase):
    group_req_id: int
    page_id: int


class NotificationType(Enum):
    REMOTE_MD = 1
    TRANSFER_NOTIFY = 2
    TRANSFER_NOTIFY_ACK = 3


@dataclass
class Notification:
    type: NotificationType
    data: Union[bytes, List[bytes]]

    def to_bytes(self):
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data):
        return pickle.loads(data)


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


class SafePageIndexScheduler:
    def __init__(self, num_pages: int):
        self.num_pages = num_pages
        self.items = list(range(num_pages))
        self.lock = Lock()
        self.cond = Condition(self.lock)

    def borrow(self, num_pages: int = 2) -> List[int]:
        if num_pages > self.num_pages:
            raise ValueError(f"Cannot borrow {num_pages} pages, only {self.num_pages} available.")

        with self.cond:
            while len(self.items) < num_pages:
                self.cond.wait()
            ret, self.items = self.items[:num_pages], self.items[num_pages:]
            return ret

    def return_(self, items: List[int]):
        with self.cond:
            self.items.extend(items)
            self.cond.notify_all()

    def current_size(self) -> int:
        with self.lock:
            return len(self.items)
