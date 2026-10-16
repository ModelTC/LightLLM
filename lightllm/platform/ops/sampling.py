from dataclasses import dataclass
from typing import Any, Callable

SamplingOp = Callable[..., Any]


@dataclass(frozen=True)
class SamplingOps:
    apply_penalty: SamplingOp
    apply_penalty_gpu_cache: SamplingOp
    apply_invalid_token_ids: SamplingOp
    token_id_counter: SamplingOp
    update_req_to_token_id_counter: SamplingOp
