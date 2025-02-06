import threading
import numpy as np

# 可以支持的最大 beam 参数上限，为了让生成的请求的group_req_id 和 sub_req_id 可以有直接的计算映射关系
# id 生成器，只会以 MAX_BEST_OF 的间隔生成id 作为 group_req_id, (sub_req_id // MAX_BEST_OF * MAX_BEST_OF) 即可
# 重新得到group_req_id

MAX_BEST_OF = 8


class ReqIDGenerator:
    def __init__(self):
        from lightllm.server.core.objs.atomic_lock import AtomicShmLock
        from lightllm.server.core.objs.shm_array import ShmArray
        from lightllm.utils.envs_utils import get_unique_server_name

        self.current_id = ShmArray(f"{get_unique_server_name()}_req_id_gen", (1,), dtype=np.int64)
        self.current_id.create_shm()
        self.current_id.arr[0] = 0
        self.lock = AtomicShmLock(f"{get_unique_server_name()}_req_id_gen_lock")

    def generate_id(self):
        with self.lock:
            id = self.current_id.arr[0]
            self.current_id.arr[0] += MAX_BEST_OF
        return id


def convert_sub_id_to_group_id(sub_req_id):
    return (sub_req_id // MAX_BEST_OF) * MAX_BEST_OF
