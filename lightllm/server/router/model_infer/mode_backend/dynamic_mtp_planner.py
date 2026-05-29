import random
import math
import numpy as np
from typing import Dict, List, Optional


class DynamicMTPPlanner:
    def __init__(
        self,
        mtp_step: int,
        ema_decay: float = 0.9,
        use_random_mode: bool = True,
        random_mode_iter_threshold: int = 100,
    ) -> None:
        self.mtp_step = mtp_step
        self.ema_decay = ema_decay

        # 记录每个请求的 verify_len 的 ema 值， 用于分布统计
        self.req_verify_len_ema = _EMAValue(self.ema_decay, init_value=float(self.mtp_step + 1))
        self.req_verify_len_second_moment_ema = _EMAValue(self.ema_decay, init_value=float((self.mtp_step + 1) ** 2))

        # 每多少个请求采用随机的方式决定 dynamic_batch_size
        self._iter = 0
        self._iter_threshold = random_mode_iter_threshold
        self._use_random_mode = use_random_mode
        random.seed(0)

        # 记录每个请求对应的单token速度记录信息
        self.req_num_to_speed_dict: Dict[int, List[_EMAValue]] = {}
        return

    def update_req_verify_len_statics(self, verify_lens: List[int]) -> None:
        for verify_len in verify_lens:
            self.req_verify_len_ema.update(float(verify_len))
            self.req_verify_len_second_moment_ema.update(float(verify_len ** 2))
        return

    def update_req_num_speed_statics(self, req_num: int, dynamic_batch_size: int, per_token_cost_ms: float) -> None:
        speed_ema_list = self._get_req_num_speed_ema_list(req_num)
        index = dynamic_batch_size - req_num
        ema_obj = speed_ema_list[index]
        ema_obj.update(per_token_cost_ms)
        return

    def get_dynamic_batch_size(self, req_num: int, original_batch_size: int) -> int:
        assert req_num * (self.mtp_step + 1) == original_batch_size
        prob_delta = 2
        # case 1 如果采用随机的方式决定 dynamic_batch_size
        self._iter += 1
        if self._use_random_mode and self._iter % self._iter_threshold == 0:
            sigma = self._get_verify_len_sigma()
            max_batch_size = min(
                req_num * (self.mtp_step + 1), int(req_num * (self.req_verify_len_ema.get() + prob_delta * sigma))
            )
            max_batch_size = max(req_num, max_batch_size)
            return random.randint(req_num, max_batch_size)

        # case 2 如果采用统计的方式决定 dynamic_batch_size, 利用统计的 ema 信息来决定
        ema_batch_size = max(req_num, int(req_num * self.req_verify_len_ema.get()))
        sigma = self._get_verify_len_sigma()
        max_batch_size = min(
            req_num * (self.mtp_step + 1), int(req_num * (self.req_verify_len_ema.get() + prob_delta * sigma))
        )
        max_batch_size = max(req_num, max_batch_size)

        start = req_num - req_num
        end = max_batch_size - req_num
        ema_index = ema_batch_size - req_num
        speed_ema_list = self._get_req_num_speed_ema_list(req_num=req_num)
        speeds = [obj.get() for obj in speed_ema_list[start : (end + 1)]]
        # 对于期望均值的位置，我们稍微降低下其数据，便于在 ema 统计的数值不够多和准确的时候，更好的选中期望位置，
        # 以获得平均性能。
        speeds[ema_index] -= 0.001
        min_index = np.argmin(speeds)
        min_cost_batch_size = min_index + req_num
        return int(min_cost_batch_size)

    def _get_req_num_speed_ema_list(self, req_num: int) -> List["_EMAValue"]:
        if req_num not in self.req_num_to_speed_dict:
            self.req_num_to_speed_dict[req_num] = [
                _EMAValue(decay=self.ema_decay, init_value=10000000.0) for _ in range(req_num * (self.mtp_step) + 1)
            ]
        return self.req_num_to_speed_dict[req_num]

    def _get_verify_len_sigma(self) -> float:
        return math.sqrt(max(0.0, self.req_verify_len_second_moment_ema.get() - self.req_verify_len_ema.get() ** 2))


class _EMAValue:
    def __init__(self, decay: float, init_value: float) -> None:
        """ """
        assert decay > 0.0 and decay < 1.0
        self.decay = decay
        self.current_decay = 0.0
        self.value = init_value

    def update(self, new_value: float) -> float:
        self.value = self.current_decay * self.value + (1.0 - self.current_decay) * new_value
        # 更新 current_decay 的值，使得 current_decay 逐渐逼近 decay 的值
        self.current_decay = min(self.decay, (self.decay + self.current_decay) / 2.0 + 0.001)
        return self.value

    def get(self) -> float:
        return self.value
