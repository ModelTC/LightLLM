from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
import numpy as np
import time


class TokenLoad:
    def __init__(self, name, dp_size) -> None:
        # 为数据并行保留的接口
        self.dp_size = dp_size
        self.shared_token_load = SharedArray(
            name,
            shape=(
                self.dp_size,
                3,
            ),
            dtype=np.float64,
        )
        # 用来保存调度需要使用到的一些信息
        self.shared_token_infos = SharedArray(
            f"{name}_ext_infos",
            shape=(
                self.dp_size,
                2,
            ),
            dtype=np.int64,
        )
        self.last_dynamic_max_load_update_time = time.time()

    # 记录系统调度器估计的峰值token使用量
    def set_estimated_peak_token_count(self, obj: int, index: int = 0):
        self.shared_token_infos.arr[index, 0] = obj
        self.last_dynamic_max_load_update_time = time.time()
        return

    def add_estimated_peak_token_count(self, value: int, index: int = 0):
        self.shared_token_infos.arr[index, 0] += value
        self.last_dynamic_max_load_update_time = time.time()
        return

    def get_estimated_peak_token_count(self, index: int = 0) -> int:
        return self.shared_token_infos.arr[index, 0]

    # 记录系统被临时固定的不能被使用的token数，主要在于 pd 分离的模式下
    # 推理系统需要在 kv 传输时临时固定一些 token， 防止调度系统估计失误，导致调度问题
    def set_frozened_token_count(self, obj: int, index: int = 0):
        self.shared_token_infos.arr[index, 1] = obj
        return

    def get_frozened_token_count(self, index: int = 0) -> int:
        return self.shared_token_infos.arr[index, 1]

    def add_frozened_token_count(self, value: int, index: int = 0):
        self.shared_token_infos.arr[index, 1] += value
        return

    # current_load 当前使用token量，估计的负载
    def set_current_load(self, value, index: int = 0):
        self.shared_token_load.arr[index, 0] = value
        return

    def get_current_load(self, index: int = 0):
        return self.shared_token_load.arr[index, 0]

    # logical_max_load 朴素估计的负载，简单将当前请求的输入和输出长度想加得到, 目前已未使用，其值与dynamic_max_load一样
    def set_logical_max_load(self, value, index: int = 0):
        self.shared_token_load.arr[index, 1] = value
        return

    def get_logical_max_load(self, index: int = 0):
        return self.shared_token_load.arr[index, 1]

    # dynamic_max_load 动态估计的最大负载，考虑请求中途退出的情况，估计的最大token使用量
    def set_dynamic_max_load(self, value, index: int = 0):
        self.shared_token_load.arr[index, 2] = value
        self.set_logical_max_load(value, index=index)
        self.last_dynamic_max_load_update_time = time.time()
        return

    def get_dynamic_max_load(self, index: int = 0):
        return self.shared_token_load.arr[index, 2]

    def need_update_dynamic_max_load(self, index: int = 0):
        # 3s 需要进行一次更新
        if time.time() - self.last_dynamic_max_load_update_time >= 3.0:
            return True
        else:
            return False
