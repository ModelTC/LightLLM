import torch
from dataclasses import dataclass
from enum import Enum
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

@dataclass
class CacheTask:
    tokens: torch.Tensor
    mode: str = None
    kv_page_indexer: torch.Tensor = None
    start_pos: torch.Tensor = 0

@dataclass
class PushState:
    state: bool

    def to_dict(self):
        return {
            "state": self.state
        }

    @staticmethod
    def from_dict(d):
        return PushState(
            state=d["state"],
        )

class HitSate(Enum):
    NONE = -1
    MEM = 0
    DISK = 1

@dataclass
class PullState:
    match_length: int
    cache_source: HitSate

    def to_dict(self):
        return {
            "match_length": self.match_length,
            "cache_source": self.cache_source.name
        }

    @staticmethod
    def from_dict(d):
        return PullState(
            match_length=d["match_length"],
            cache_source=HitSate[d["cache_source"]]
        )