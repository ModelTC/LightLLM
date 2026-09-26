"""EPLB 后台线程任务的公共生命周期。"""

import os
import threading

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class EPLBAsyncTask:
    """提供单次后台任务统一的启动、完成和失败处理。"""

    def __init__(self, *, thread_name: str) -> None:
        self.status = "idle"
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    def start(self) -> None:
        """启动后台任务；同一个任务对象只能启动一次。"""
        assert self.status == "idle", f"{type(self).__name__} has already been started"
        self.status = "running"
        self._thread.start()

    def is_finished(self) -> bool:
        """返回后台任务是否已经成功完成。"""
        return self.status == "succeeded"

    def _run(self) -> None:
        try:
            self.execute()
        except BaseException:
            logger.exception(f"{type(self).__name__} failed")
            os._exit(1)
        else:
            self.status = "succeeded"

    def execute(self) -> None:
        """执行子类定义的具体后台任务。"""
        raise NotImplementedError
