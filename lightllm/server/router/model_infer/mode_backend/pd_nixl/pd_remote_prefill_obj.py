from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import List, Union, Optional, Any
from threading import Lock

from lightllm.utils.log_utils import init_logger
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import RemotePrefillServerInfo

logger = init_logger(__name__)

try:
    from nixl._api import nixlBind, nixl_prepped_dlist_handle

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
class PrefillRequest(RemoteRequest):
    decode_id: int
    data: RemotePrefillRequest


@dataclass
class KVMoveRequest:
    group_req_id: int
    token_ids: List[int]


@dataclass
class RemoteAgent:
    name: str
    num_tokens: int
    kv_mem_desc: nixlBind.nixlRegDList
    kv_xfer_handles: nixl_prepped_dlist_handle


@dataclass
class RemotePrefillStatus:
    group_req_id: int
    status: int

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
            return len(self._data)

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
