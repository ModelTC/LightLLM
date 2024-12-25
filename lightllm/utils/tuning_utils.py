import torch
import multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing.pool import util, worker
from typing import Any, Dict
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@staticmethod
def fix_repopulate_pool_static(
    ctx, Process, processes, pool, inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception
):
    for i in range(processes - len(pool)):
        w = Process(
            ctx, target=worker, args=(inqueue, outqueue, initializer, initargs, maxtasksperchild, wrap_exception)
        )
        w.name = w.name.replace("Process", "PoolWorker")
        w.daemon = False  # modify to False
        w.start()
        pool.append(w)
        util.debug("added worker")


def run_func(func, args):
    return func(**args)


def mp_tuning(func, args: Dict[str, Any]):
    # 修复 pool 中的进程无法启动子进程进行 kernel tuning 的问题
    Pool._repopulate_pool_static = fix_repopulate_pool_static
    device_count = torch.cuda.device_count()

    with mp.Pool(processes=device_count) as pool:
        tasks = []
        for device_id in range(device_count):
            t = {
                "device_id": device_id,
                "device_count": device_count,
            }
            t.update(args)
            tasks.append((func, t))

        results = pool.starmap(run_func, tasks)

    best_config, best_cost_time = None, 100000000000000000000
    for _config, _cost_time in results:
        if _cost_time is not None:
            if _cost_time < best_cost_time:
                best_cost_time = _cost_time
                best_config = _config

    logger.info(f"best config {best_config} best cost time {best_cost_time}")
    return best_config
