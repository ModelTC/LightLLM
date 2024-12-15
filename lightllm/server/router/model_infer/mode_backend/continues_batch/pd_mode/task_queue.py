import threading
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class TaskQueue:
    def __init__(self, get_func, fail_func):
        self.lock = threading.Lock()
        self.datas = []
        self.get_func = get_func
        self.fail_func = fail_func
        self.has_error = False

    def size(self):
        return len(self.datas)

    def put(self, obj):
        if self.has_error:
            raise Exception("has error")

        with self.lock:
            self.datas.append(obj)

    def put_list(self, objs):
        if self.has_error:
            raise Exception("has error")

        with self.lock:
            self.datas.extend(objs)

    def get_tasks(self, log_tag=None):
        with self.lock:
            ans = self.get_func(self.datas)
            self.datas = self.datas[len(ans) :]
        if len(self.datas) != 0:
            logger.info(f"queue {log_tag} left size: {len(self.datas)}")
        return ans

    def clear_tasks(self):
        with self.lock:
            if len(self.datas) != 0:
                for obj in self.datas:
                    self.fail_func(obj)
            self.datas = []
        return
