import threading
from collections import deque
from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.utils.envs_utils import get_env_start_args


class MultiLevelCacheManager(object):
    def __init__(self, backend):
        self.args = get_env_start_args()
        from .base_backend import ModeBackend

        self.backend: ModeBackend = backend
        if self.args.enable_cpu_cache:
            self.cpu_cache_handle_queue = deque()
            self.cpu_cache_client = CpuKvCacheClient(init_shm_data=False)
            self.cpu_cache_thread = threading.Thread(target=self.cpu_cache_handle_loop, daemon=True)
            self.cpu_cache_thread.start()

    def cpu_cache_handle_loop(self):
        pass
