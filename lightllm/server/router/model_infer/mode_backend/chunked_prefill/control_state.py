from enum import Enum
from typing import List
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.infer_batch import InferReq

class ControlState:

    def __init__(self):
        self.is_aggressive_schedule = not get_env_start_args().disable_aggressive_schedule

        # 非激进调度参数
        self.decode_max_step = max(1, get_env_start_args().router_max_wait_tokens)
        self.left_decode_num = self.decode_max_step

        self.step_count = 0


    def select_run_way(self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]) -> "RunWay":
        """
        判断决策运行方式：
        返回值: RunWay
        """
        self.step_count += 1
        if self.is_aggressive_schedule:
            return self._agressive_way(prefill_reqs=prefill_reqs,
                                       decode_reqs=decode_reqs)
        else:
            return self._normal_way(prefill_reqs=prefill_reqs,
                                    decode_reqs=decode_reqs)

    def _agressive_way(self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]):
        if prefill_reqs:
            return RunWay.PREFILL
        if decode_reqs:
            return RunWay.DECODE
        return RunWay.PASS
    
    def _normal_way(self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]):
        if decode_reqs:
            if self.left_decode_num > 0:
                self.left_decode_num -= 1
                return RunWay.DECODE
            else:
                if prefill_reqs:
                    self.left_decode_num = self.decode_max_step
                    return RunWay.PREFILL
                else:
                    return RunWay.DECODE
        else:
            if prefill_reqs:
                self.left_decode_num = self.decode_max_step
                return RunWay.PREFILL
            else:
                return RunWay.PASS
            
    def try_recover_paused_reqs(self) -> bool:
        return self.step_count % 100 == 0

        

class RunWay(Enum):
    PREFILL = 1
    DECODE = 2
    PASS = 3

    def is_prefill(self):
        return self == RunWay.PREFILL

    def is_decode(self):
        return self == RunWay.DECODE

    def is_pass(self):
        return self == RunWay.PASS