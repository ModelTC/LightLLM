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
