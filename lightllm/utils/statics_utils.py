import time
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MovingAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self._last_logged_count = 0
        self.last_time = time.time()

    def add(self, value):
        self.total += value
        self.count += 1

    def average(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def print_log(self, log_str):
        if self.count > self._last_logged_count and time.time() - self.last_time >= 30:
            logger.debug("event=request.latency kind=%r samples=%s mean_ms=%.3f", log_str, self.count, self.average())
            self._last_logged_count = self.count
            self.last_time = time.time()
