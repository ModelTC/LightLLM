from dataclasses import dataclass
from typing import List, Optional, Any, Union
from lightllm.utils.torch_memory_saver_utils import MemoryTag


@dataclass
class AbortReq:
    # 外部调用传入，等同内部的 group_req_id
    request_id: Optional[int] = None
    abort_all: bool = False


def _normalize_memory_tags(tags):
    if tags is None:
        return None
    return [tag if isinstance(tag, MemoryTag) else MemoryTag(tag) for tag in tags]


@dataclass
class FlushCacheReq:
    pass


@dataclass
class ReleaseMemoryReq:
    tags: Optional[List[MemoryTag]] = None

    def __post_init__(self):
        self.tags = _normalize_memory_tags(self.tags)


@dataclass
class ResumeMemoryReq:
    tags: Optional[List[MemoryTag]] = None

    def __post_init__(self):
        self.tags = _normalize_memory_tags(self.tags)


@dataclass
class GeneralHttpToModelRpcReq:
    func_name: str
    func_args: Optional[Any] = None


@dataclass
class GeneralModelToHttpRpcRsp:
    success: bool
    msg: Optional[str]
    func_name: str
    func_rsp: Optional[Any] = None


@dataclass
class InitWeightsUpdateGroupReq:
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str = "weight_update_group"
    backend: str = "nccl"


@dataclass
class DestroyWeightsUpdateGroupReq:
    group_name: str = "weight_update_group"


@dataclass
class UpdateWeightsFromDistributedReq:
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromTensorReq:
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[Union[str, bytes]]
    load_format: Optional[str] = None
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReq:
    ipc_handle: Optional[Union[str, dict]] = None
    use_shm: bool = False
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None
