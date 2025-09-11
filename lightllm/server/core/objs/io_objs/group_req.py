from dataclasses import dataclass
from lightllm.server.multimodal_params import MultimodalParams
from typing import List
from ..req import Req


@dataclass
class GroupReqIndexes:
    group_req_id: int
    multimodal_params: MultimodalParams
    shm_req_indexes: List[int]
    time_mark: float


@dataclass
class GroupReqObjs:
    group_req_id: int
    multimodal_params: MultimodalParams
    shm_req_objs: List[Req]
    time_mark: float

    def to_group_req_index(self):
        return GroupReqIndexes(
            group_req_id=self.group_req_id,
            multimodal_params=self.multimodal_params,
            shm_req_indexes=[req.index_in_shm_mem for req in self.shm_req_objs],
            time_mark=self.time_mark,
        )


@dataclass
class AbortedReqCmd:
    req_id: int


@dataclass
class StopStrMatchedReqCmd:
    req_id: int
