import threading
import time
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Watchdog:
    def __init__(self, timeout):
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self.running = True

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            time.sleep(2)
            if time.time() - self.last_heartbeat > self.timeout:
                logger.error("Watchdog: Timeout! Task is not responding.")
                self.handle_timeout()

    def handle_timeout(self):
        logger.error("Watchdog: time out to exit")
        import sys

        sys.exit(-1)

    def stop(self):
        self.running = False
        self.thread.join()

    def heartbeat(self):
        self.last_heartbeat = time.time()
