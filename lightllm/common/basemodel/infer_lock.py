# 这不是一个很好的设计但是不是很好找到更好更简单对架构入侵更小的实现方法。
# 这个地方声明的锁和计数，主要是用来解决在 PD 分离模式下，kv_move_manager 进程中会出现
# 通过rpyc调用操作 radix cache 和 mem_manager 中的数据的问题，这可能导致严重的数据同步
# 问题，主要原因是各个tp的推理进程运行到的位置节点并没有严格的保证，导致radix cache 和
# mem manager 中的数据出现各个进程间不一致的问题。
# 下面的实现中，通过一个锁和计数对象, 配合使用的方式，来解决这个问题。
from dataclasses import dataclass
import numpy as np
import threading
from lightllm.server.router.dynamic_prompt.shared_arr import SharedArray
import torch.distributed as dist
import time
import torch.multiprocessing as mp
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class InferStateLock:
    def __init__(self, name):
        self.infer_lock = threading.Lock()
        # 默认开 128 tp 的空间, 现在应该没什么卡能开这么大的tp 吧
        self.lock_tp_infos = SharedArray(f"{name}_lock_tp_infos", shape=(129,), dtype=np.int64)
        self.lock_tp_infos.arr[:] = 0
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()

    def add_cur_mark(self):
        self.lock_tp_infos.arr[self.rank_id] += 1

    def get_cur_mark(self):
        return self.lock_tp_infos.arr[self.rank_id]

    def get_max_mark_in_group(self):
        return np.max(self.lock_tp_infos.arr[0 : self.world_size])

    def judge_cur_mark_equal_max_mark_in_group(self):
        return self.get_cur_mark() == self.get_max_mark_in_group()

    def judge_mark_in_group_all_same(self):
        marks = self.lock_tp_infos.arr[0 : self.world_size]
        return bool(np.all(marks == marks[0]))

    def acquire_lock_and_update_cur_mark(self):
        self.infer_lock.acquire()
        self.add_cur_mark()

    def release_lock(self):
        self.infer_lock.release()

    def set_group_wait_mark(self):
        if self.rank_id == 0:
            self.lock_tp_infos.arr[-1] = 1

    def unset_group_wait_mark(self):
        if self.rank_id == 0:
            self.lock_tp_infos.arr[-1] = 0

    def get_group_wait_mark(self):
        return self.lock_tp_infos.arr[-1]


@dataclass
class G_Infer_Lock:
    obj: InferStateLock = None
    dp_size: int = None

    def acquire(self):
        if self.obj is not None:
            # 当遇到有同步请求的时候，同时自己的mark已经是最大的mark的时候，就在这里休眠，
            # 不去竞争锁, 因为 wait_mark == 1 的时候， 说明acquire_lock_until_ready被调用，
            # 有推理进程在申请同步点操作
            while self.obj.get_group_wait_mark() == 1 and self.obj.judge_cur_mark_equal_max_mark_in_group():
                time.sleep(0)

            self.obj.acquire_lock_and_update_cur_mark()

    def release(self):
        if self.obj is not None:
            self.obj.release_lock()


# 后续由 backend 对象来对obj进行初始化赋值，方便进行全局调用
g_infer_state_lock = G_Infer_Lock()


# 下面两个函数需要配对使用
def acquire_lock_until_ready(nccl_group):
    # 在 deepseekv2 的tp dp 混合运行模式下, 不需要多个推理进程间做协调同步
    # 所以直接加锁，解锁即可
    if g_infer_state_lock.dp_size != 1:
        g_infer_state_lock.obj.infer_lock.acquire()
        return

    g_infer_state_lock.obj.set_group_wait_mark()
    while True:
        g_infer_state_lock.obj.infer_lock.acquire()
        dist.barrier(nccl_group)
        judge_ans = g_infer_state_lock.obj.judge_mark_in_group_all_same()
        dist.barrier(nccl_group)

        if judge_ans is not True:
            # 释放锁进行重试
            g_infer_state_lock.obj.infer_lock.release()
            time.sleep(0.001)
            logger.info("wait get locks sleep 1ms")
        else:
            break

    g_infer_state_lock.obj.unset_group_wait_mark()
    return


def release_acquired_lock():
    g_infer_state_lock.obj.infer_lock.release()


@dataclass
class G_Router_Lock:
    """
    保护pd分离模式下, 一些数据的操作。
    """

    obj = None  # 进程锁对象

    def acquire(self):
        if self.obj is not None:
            self.obj.acquire()

    def release(self):
        if self.obj is not None:
            self.obj.release()


g_router_lock = G_Router_Lock()
