import time


class TimeChecker:
    def __init__(self, threshold):
        self.threshold = threshold
        self.last_checked = time.time()

    def has_exceeded(self):
        current_time = time.time()
        if (current_time - self.last_checked) > self.threshold:
            self._reset()
            return True
        return False

    def _reset(self):
        self.last_checked = time.time()
