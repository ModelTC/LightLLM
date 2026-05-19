from typing import Dict, List, Optional


class DynamicMTPPlanner:
    def __init__(
        self,
        mtp_step: int,
        ema_decay: float = 0.9,
        confidence_k: float = 1.0,
    ) -> None:
        self.mtp_step = mtp_step
        self.confidence_k = confidence_k
        # 记录每个请求的 accept_len 的 ema 值， 用于分布统计
        self.req_accept_len_ema = _EMAValue(ema_decay, init_value=float(self.mtp_step + 1))
        self.req_accept_len_second_moment_ema = _EMAValue(ema_decay, init_value=float((self.mtp_step + 1) ** 2))

        # 记录每个请求对应的单token速度记录信息
        self.req_num_to_speed_dict: Dict[int, List[_EMAValue]] = {}

    def update_req_accept_len_statics(self, accept_lens: List[int]) -> None:
        for accept_len in accept_lens:
            self.req_accept_len_ema.update(float(accept_len))
            self.req_accept_len_second_moment_ema.update(float(accept_len ** 2))
        return

    def update_req_num_speed_statics(self, req_num: int, dynamic_batch_size: int, per_token_cost_ms: float) -> None:
        speed_ema_list = self._get_req_num_speed_ema_list(req_num)
        index = dynamic_batch_size - req_num
        ema_obj = speed_ema_list[index]
        ema_obj.update(per_token_cost_ms)
        return

    def get_dynamic_batch_size(self, req_num: int, original_batch_size: int) -> int:
        assert req_num * (self.mtp_step + 1) == original_batch_size

        mean = self.req_accept_len_ema.get()
        return max(req_num, int(mean))

    def _get_req_num_speed_ema_list(self, req_num: int) -> List["_EMAValue"]:
        if req_num not in self.req_num_to_speed_dict:
            self.req_num_to_speed_dict[req_num] = [
                _EMAValue(self.ema_decay, init_value=10000000.0) for _ in range(req_num * (self.mtp_step))
            ]
        return self.req_num_to_speed_dict[req_num]


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
