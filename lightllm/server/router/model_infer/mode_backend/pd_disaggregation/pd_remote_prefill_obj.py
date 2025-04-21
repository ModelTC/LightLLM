from dataclasses import dataclass
from enum import Enum
from typing import List, Union

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
    local_cached_len: int # will skip transfer
    token_ids: List[int] # mem cache indexes


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