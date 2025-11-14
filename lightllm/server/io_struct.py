from dataclasses import dataclass


@dataclass
class AbortReq:
    request_id: int = -1
    abort_all: bool = False
