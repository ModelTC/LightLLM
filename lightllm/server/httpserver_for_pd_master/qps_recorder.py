import time
from collections import deque
from threading import Lock
from typing import Deque, Optional

from lightllm.utils.envs_utils import get_pd_request_limit_max_allowed_request_count_seconds


class QPSRecorder:
    """根据最近完成的请求计算 PD Master 的动态 QPS。"""

    def __init__(self, args, ema_alpha: float = 0.1):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in the range (0, 1]")

        self.args = args
        self.ema_alpha = float(ema_alpha)
        # 保存最近 64 个请求的完成时间。64 个时间点之间包含 63 个完成间隔。
        self._finished_timestamps: Deque[float] = deque(maxlen=64)
        self._qps = 0.0
        self._initialized = False
        self._last_qps_update_time: Optional[float] = None
        self._lock = Lock()

    def mark_one_req_finish(self) -> None:
        """记录一个请求完成事件，并在样本充足时更新全局 QPS。"""
        finished_time = time.monotonic()
        with self._lock:
            self._finished_timestamps.append(finished_time)
            self._update_qps()

    def get_qps(self) -> float:
        """返回经过 EMA 平滑后的全局 QPS。"""
        with self._lock:
            if self._last_qps_update_time is not None:
                current_time = time.monotonic()
                if current_time - self._last_qps_update_time > 30:
                    self._update_qps()
            return self._qps

    def get_max_allowed_request_count(self, default_max_allowed_request_count: int) -> int:
        """返回 PD Master 最大允许进入请求数。

        QPS 尚未初始化或完成样本不足时，使用调用方传入的 Decode 节点总并发容量；
        样本充足后使用动态 QPS 估算值。
        """
        with self._lock:
            if not self._initialized:
                return default_max_allowed_request_count

        # PD Master 统计完整 PD 请求，按统一的平均整包时长配置估算在途请求数。
        return int(self.get_qps() * get_pd_request_limit_max_allowed_request_count_seconds()) + 6

    def _update_qps(self) -> None:
        if len(self._finished_timestamps) < self._finished_timestamps.maxlen:
            return

        current_time = time.monotonic()
        elapsed_time = current_time - self._finished_timestamps[0]
        if elapsed_time <= 0:
            return

        average_qps = (len(self._finished_timestamps) - 1) / elapsed_time
        if not self._initialized:
            self._qps = average_qps
            self._initialized = True
        else:
            self._qps = self.ema_alpha * average_qps + (1 - self.ema_alpha) * self._qps
        self._last_qps_update_time = current_time
