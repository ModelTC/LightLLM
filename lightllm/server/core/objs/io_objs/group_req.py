from dataclasses import dataclass
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.core.objs.sampling_params import SamplingParams
from typing import List
from ..req import Req


@dataclass
class AbortedReqCmd:
    req_id: int


@dataclass
class StopStrMatchedReqCmd:
    req_id: int
