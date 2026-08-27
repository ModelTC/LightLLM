from enum import Enum
from lightllm.utils.envs_utils import get_env_start_args
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
        has_prefill: bool,
        has_decode: bool,
    ) -> "RunWay":
        """
        判断决策运行方式：
        返回值: RunWay
        """
        self.step_count += 1
        if self.is_aggressive_schedule:
            return self._agressive_way(
                has_prefill=has_prefill,
                has_decode=has_decode,
            )
        else:
            return self._normal_way(
                has_prefill=has_prefill,
                has_decode=has_decode,
            )

    def _agressive_way(
        self,
        has_prefill: bool,
        has_decode: bool,
    ):
        if has_prefill:
            return RunWay.PREFILL
        if has_decode:
            return RunWay.DECODE
        return RunWay.PASS

    def _normal_way(
        self,
        has_prefill: bool,
        has_decode: bool,
    ):
        if self.left_decode_num > 0 and has_decode:
            self.left_decode_num -= 1
            return RunWay.DECODE

        if has_prefill:
            # prefill 一次允许进行几次 decode 操作。
            self.left_decode_num = self.decode_max_step
            return RunWay.PREFILL
        else:
            if has_decode:
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
