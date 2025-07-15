from dataclasses import dataclass
from typing import List

@dataclass
class ShmReqInfo:
    request_id: int
    shm_req_index: int

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "shm_req_index": self.shm_req_index
        }

    @staticmethod
    def from_dict(d):
        return ShmReqInfo(
            request_id=d["request_id"],
            shm_req_index=d["shm_req_index"]
        )

@dataclass
class GroupReqInfo:
    group_req_id: int
    shm_req_indexes: List[int]

    def to_dict(self):
        return {
            "group_req_id": self.group_req_id,
            "shm_req_indexes": self.shm_req_indexes
        }
    
    @staticmethod
    def from_dict(d):
        return GroupReqInfo(
            group_req_id=d["group_req_id"],
            shm_req_indexes=d["shm_req_indexes"]
        )