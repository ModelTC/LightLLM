from abc import ABC
from dataclasses import dataclass
from lightllm.server.core.objs.req import Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from typing import List, Optional, Any, Union


@dataclass
class BaseReq(ABC):
    def get_req_to_next_node(self):
        return self

    def get_req_to_next_module(self):
        return self

@dataclass
class BaseRsp(ABC):
    success: bool
    msg: Optional[str]

# for next node
@dataclass
class GenerateReqMeta(BaseReq):
    prompt: str
    sampling_params: SamplingParams
    multimodal_params: MultimodalParams


# for next module
@dataclass
class GenerateReqIndex(BaseReq):
    group_req_id: int
    multimodal_params: MultimodalParams
    shm_req_indexes: List[int]
    time_mark: float


@dataclass
class GenerateReq(BaseReq):
    group_req_id: int
    prompt: str
    sampling_params: SamplingParams
    multimodal_params: MultimodalParams
    shm_req_objs: List[Req]
    time_mark: float

    def get_req_to_next_module(self):
        # 已经完成跨节点转发，可以释放图片原始资源
        self.multimodal_params.free()
        return GenerateReqIndex(
            group_req_id=self.group_req_id,
            multimodal_params=self.multimodal_params,
            shm_req_indexes=[req.index_in_shm_mem for req in self.shm_req_objs],
            time_mark=self.time_mark,
        )

    def get_req_to_next_node(self):
        return GenerateReqMeta(
            prompt=self.prompt,
            sampling_params=self.sampling_params,
            multimodal_params=self.multimodal_params,
        )


@dataclass
class GenerateResp(BaseReq):
    pass


@dataclass
class FlushCacheReq(BaseReq):
    pass


@dataclass
class FlushCacheResp(BaseReq):
    success: bool


@dataclass
class AbortReq(BaseReq):
    # 外部调用传入，等同内部的 group_req_id
    request_id: int = None
    abort_all: bool = False


@dataclass
class GeneralHttpToModelRpcReq(BaseReq):
    func_name: str
    func_args: Optional[Any] = None

@dataclass
class GeneralModelToHttpRpcRsp(BaseRsp):
    func_name: str
    func_rsp: Optional[Any] = None

@dataclass
class InitWeightsUpdateGroupReq(BaseReq):
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"

@dataclass
class InitWeightsUpdateGroupRsp(BaseRsp):
    pass

@dataclass
class DestroyWeightsUpdateGroupReq(BaseReq):
    group_name: str = "weight_update_group"

@dataclass
class DestroyWeightsUpdateGroupRsp(BaseRsp):
    pass

@dataclass
class UpdateWeightsFromDistributedReq(BaseReq):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    # The group name
    group_name: str = "weight_update_group"
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None

@dataclass
class UpdateWeightsFromDistributedRsp(BaseRsp):
    pass


@dataclass
class UpdateWeightsFromTensorReq(BaseReq):
    """Update model weights from tensor input.

    - Tensors are serialized for transmission
    - Data is structured in JSON for easy transmission over HTTP
    """

    serialized_named_tensors: List[Union[str, bytes]]
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None

@dataclass
class UpdateWeightsFromTensorRsp(BaseRsp):
    pass