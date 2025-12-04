import numpy as np
from enum import Enum
from typing import List
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.infer_batch import InferReq
from ..base_backend import ModeBackend


class DPControlState:
    def __init__(self, backend: ModeBackend):
        self.backend = backend
        self.is_aggressive_schedule = not get_env_start_args().disable_aggressive_schedule

        # 非激进调度参数
        self.decode_max_step = max(0, get_env_start_args().router_max_wait_tokens)
        self.left_decode_num = self.decode_max_step

        self.step_count = 0
        return

    def select_run_way(
        self,
        dp_prefill_req_nums: np.ndarray,
        dp_decode_req_nums: np.ndarray,
        prefill_reqs: List[InferReq],
        decode_reqs: List[InferReq],
    ) -> "RunWay":
        """
        判断决策运行方式：
        返回值: RunWay
        """
        self.step_count += 1
        if self.is_aggressive_schedule:
            return self._agressive_way(
                dp_prefill_req_nums=dp_prefill_req_nums,
                dp_decode_req_nums=dp_decode_req_nums,
                prefill_reqs=prefill_reqs,
                decode_reqs=decode_reqs,
            )
        else:
            return self._normal_way(
                dp_prefill_req_nums=dp_prefill_req_nums,
                dp_decode_req_nums=dp_decode_req_nums,
                prefill_reqs=prefill_reqs,
                decode_reqs=decode_reqs,
            )

    def _agressive_way(
        self,
        dp_prefill_req_nums: np.ndarray,
        dp_decode_req_nums: np.ndarray,
        prefill_reqs: List[InferReq],
        decode_reqs: List[InferReq],
    ):
        max_prefill_num = np.max(dp_prefill_req_nums)
        if max_prefill_num > 0:
            return RunWay.PREFILL
        max_decode_num = np.max(dp_decode_req_nums)
        if max_decode_num > 0:
            return RunWay.DECODE
        return RunWay.PASS

    def _normal_way(
        self,
        dp_prefill_req_nums: np.ndarray,
        dp_decode_req_nums: np.ndarray,
        prefill_reqs: List[InferReq],
        decode_reqs: List[InferReq],
    ):
        # use_ratio = np.count_nonzero(dp_prefill_req_nums) / dp_prefill_req_nums.shape[0]
        max_decode_num = np.max(dp_decode_req_nums)
        max_prefill_num = np.max(dp_prefill_req_nums)

        if self.left_decode_num > 0 and max_decode_num > 0:
            self.left_decode_num -= 1
            return RunWay.DECODE

        if max_prefill_num > 0:
            # prefill 一次允许进行几次 decode 操作。
            self.left_decode_num = self.decode_max_step
            return RunWay.PREFILL
        else:
            if max_decode_num > 0:
                return RunWay.DECODE
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
