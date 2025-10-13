import os
import sys
import torch
import multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing.pool import util, worker
from typing import Callable, Any, Dict, List
from lightllm.utils.log_utils import init_logger
from lightllm.utils.watchdog_utils import Watchdog

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

    logger.info(f"args: {args} best config {best_config} best cost time {best_cost_time}")
    return best_config


def set_seed():
    import torch
    import random
    import numpy as np

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def workder(
    test_func: Callable[..., float],
    test_configs: List[Dict[str, Any]],
    test_kwargs: Dict[str, Any],
    queue: mp.Queue,
):
    dog = Watchdog(timeout=10)
    dog.start()

    try:
        for cfg in test_configs:
            cost_time = test_func(**test_kwargs, **cfg)
            dog.heartbeat()
            queue.put(cost_time)
    except Exception as ex:
        logger.error(f"{str(ex)}  config: {cfg}")
        sys.exit(-1)


def tuning_configs(device_id, device_count, **configs):
    test_func = configs.pop("test_func")
    test_kwargs = configs.pop("test_func_args")
    get_test_configs_func = configs.pop("get_test_configs_func")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    best_config, best_cost_time = None, float("inf")
    queue = mp.Queue()
    test_configs_list = []
    for t_config in get_test_configs_func(device_id, device_count, **test_kwargs):
        test_configs_list.append(t_config)
        if len(test_configs_list) < 64:
            continue

        p = mp.Process(
            target=workder,
            args=(test_func, test_configs_list, test_kwargs, queue),
        )
        p.start()
        p.join()

        while len(test_configs_list) > 0:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs_list[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs_list[0]
                    best_cost_time = cost_time
                    logger.info(f"current best: {best_config}, cost_time: {best_cost_time}")
                del test_configs_list[0]
            except:
                logger.info(f"current best: {best_config}, cost_time: {best_cost_time}")
                del test_configs_list[0]
                break

    while len(test_configs_list) > 0:
        p = mp.Process(
            target=workder,
            args=(test_func, test_configs_list, test_kwargs, queue),
        )
        p.start()
        p.join()

        while len(test_configs_list) > 0:
            try:
                cost_time = queue.get_nowait()
                logger.info(f"get {test_configs_list[0]} cost_time: {cost_time}")
                if cost_time < best_cost_time:
                    best_config = test_configs_list[0]
                    best_cost_time = cost_time
                    logger.info(f"current best: {best_config}, cost_time: {best_cost_time}")
                del test_configs_list[0]
            except:
                logger.info(f"current best: {best_config}, cost_time: {best_cost_time}")
                del test_configs_list[0]
                break

    logger.info(f"Final best config: {best_config}, cost_time: {best_cost_time}")
    return best_config, best_cost_time
