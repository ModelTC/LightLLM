from dataclasses import dataclass
from typing import Callable

from lightllm.platform import DeviceRuntime


@dataclass(frozen=True, kw_only=True)
class BackendEnv:
    runtime: DeviceRuntime
    hold_req_id: int
    draft_step: int
    max_seq_len: int
    tp_size: int
    alloc_tensor: Callable
    state_slots: int = 1
    tp_rank: int = 0
    layer_ids: tuple[int, ...] = ()
    prefill_graph: bool = False
    max_prefill_tokens: int = 0
    decode_only: bool = False
