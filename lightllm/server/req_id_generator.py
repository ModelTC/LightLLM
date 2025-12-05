import os
import psutil
import sys
import time
import requests
import numpy as np
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 可以支持的最大 beam 参数上限，为了让生成的请求的group_req_id 和 sub_req_id 可以有直接的计算映射关系
# id 生成器，只会以 MAX_BEST_OF 的间隔生成id 作为 group_req_id, (sub_req_id // MAX_BEST_OF * MAX_BEST_OF) 即可
# 重新得到group_req_id

MAX_BEST_OF = 8


class ReqIDGenerator:
    def __init__(self):
        from lightllm.server.core.objs.atomic_lock import AtomicShmLock
        from lightllm.server.core.objs.shm_array import ShmArray
        from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args

        self.args = get_env_start_args()
        self.use_config_server = (
            self.args.config_server_host and self.args.config_server_port and self.args.run_mode == "pd_master"
        )
        self.current_id = ShmArray(f"{get_unique_server_name()}_req_id_gen", (2,), dtype=np.int64)
        self.current_id.create_shm()
        self.current_id.arr[0] = 0
        self.current_id.arr[1] = 0
        self.lock = AtomicShmLock(f"{get_unique_server_name()}_req_id_gen_lock")
        self._wait_all_workers_ready()
        logger.info("ReqIDGenerator init finished")

    def _wait_all_workers_ready(self):
        from lightllm.utils.envs_utils import get_unique_server_name
        from lightllm.server.core.objs.shm_array import ShmArray

        _sync_shm = ShmArray(
            f"{get_unique_server_name()}_httpworker_start_sync", (self.args.httpserver_workers,), dtype=np.int64
        )
        _sync_shm.create_shm()
        # 等待所有 httpserver 的 worker 启动完成，防止重新初始化对应的请求id 对应的shm
        try_count = 0
        while len(_find_sibling_processes()) + 1 != self.args.httpserver_workers:
            time.sleep(0.1)
            try_count += 1
            if try_count > 120:
                logger.error("wait all httpserver workers start failed")
                sys.exit(-1)
            else:
                continue

        cur_p_id = os.getpid()
        pids = _find_sibling_processes()
        pids.append(cur_p_id)
        assert len(pids) == self.args.httpserver_workers
        pids = sorted(pids)
        index = pids.index(cur_p_id)
        _sync_shm.arr[index] = cur_p_id
        try_count = 0
        while not all(a == b for a, b in zip(pids, _sync_shm.arr)):
            time.sleep(0.1)
            try_count += 1
            if try_count > 120:
                logger.error("wait all httpserver workers start failed 1")
                sys.exit(-1)
            else:
                continue

    def _check_and_set_new_id_range(self):
        need_update_range = self.current_id.arr[0] + MAX_BEST_OF >= self.current_id.arr[1]
        if need_update_range:
            if not self.use_config_server:
                self.current_id.arr[0] = MAX_BEST_OF
                self.current_id.arr[1] = np.iinfo(np.int64).max
            else:
                while True:
                    try:
                        config_server_ip_port = f"{self.args.config_server_host}:{self.args.config_server_port}"
                        url = f"http://{config_server_ip_port}/allocate_global_unique_id_range"
                        response = requests.get(url)
                        if response.status_code == 200:
                            id_range = response.json()
                            logger.info(f"get new id range {id_range}")
                            # 保证id满足倍乘关系
                            self.current_id.arr[0] = (id_range["start_id"] // MAX_BEST_OF + 1) * MAX_BEST_OF
                            self.current_id.arr[1] = id_range["end_id"]
                            assert (
                                self.current_id.arr[0] + MAX_BEST_OF < self.current_id.arr[1]
                            ), f"get id range error {self.current_id.arr[0]} {self.current_id.arr[1]}"
                            return
                        else:
                            raise RuntimeError(f"Failed to fetch ID range from config server: {response.status_code}")
                    except BaseException as e:
                        logger.exception(str(e))
                        time.sleep(3)

    def generate_id(self):
        with self.lock:
            self._check_and_set_new_id_range()
            id = self.current_id.arr[0]
            self.current_id.arr[0] += MAX_BEST_OF
        return id


def convert_sub_id_to_group_id(sub_req_id):
    return (sub_req_id // MAX_BEST_OF) * MAX_BEST_OF


def _find_sibling_processes():
    # 获取当前进程的 PID
    current_pid = os.getpid()

    # 获取当前进程的信息
    current_process = psutil.Process(current_pid)

    # 获取当前进程的父进程
    parent_process = current_process.parent()

    if parent_process is None:
        logger.error("Current process has no parent.")
        return []

    # 查找兄弟进程
    sibling_processes = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            # 检查是否是兄弟进程（同一父进程且不是当前进程）
            if proc.pid != current_pid and proc.ppid() == parent_process.pid:
                sibling_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return [proc.pid for proc in sibling_processes]
