from abc import ABC
from dataclasses import dataclass
from lightllm.server.core.objs.req import Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from typing import List, Optional
from lightllm.utils.torch_memory_saver_utils import MemoryTag


@dataclass
class BaseReq(ABC):
    def get_req_to_next_node(self):
        return self

    def get_req_to_next_module(self):
        return self


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
class ReleaseMemoryReq(BaseReq):
    tags: Optional[List[MemoryTag]] = None


@dataclass
class ReleaseMemoryResp(BaseReq):
    success: bool


@dataclass
class ResumeMemoryReq(BaseReq):
    tags: Optional[List[MemoryTag]] = None


@dataclass
class ResumeMemoryResp(BaseReq):
    success: bool
