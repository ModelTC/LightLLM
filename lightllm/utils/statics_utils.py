import time
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MovingAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.last_time = time.time()

    def add(self, value):
        self.total += value
        self.count += 1

    def average(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def print_log(self, log_str):
        if time.time() - self.last_time >= 30:
            logger.info(f"{log_str}: {self.average()} ms")
            self.last_time = time.time()
