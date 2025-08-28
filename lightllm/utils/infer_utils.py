import torch
import torch.distributed as dist

import time
from typing import Callable

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

is_show_cost_time = False


def mark_cost_time(func_name):
    def inner_func(func):
        def time_func(*args, **kwargs):
            if dist.get_rank() in [0, 1] and is_show_cost_time:
                torch.cuda.synchronize()
                start_time = time.time()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                logger.debug(f"{func_name} cost time: {(time.time() - start_time) * 1000}")
                return ans
            else:
                torch.cuda.synchronize()
                ans = func(*args, **kwargs)
                torch.cuda.synchronize()
                return ans

        return time_func

    return inner_func


time_mark = {}


def mark_start(key):
    torch.cuda.synchronize()
    global time_mark
    time_mark[key] = time.time()
    return


def mark_end(key, print_min_cost=0.0):
    torch.cuda.synchronize()
    global time_mark
    cost_time = (time.time() - time_mark[key]) * 1000
    if cost_time > print_min_cost:
        logger.debug(f"cost {key}: {cost_time}")


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    logger.debug(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def benchmark_time(func: Callable, *args, warmup: int = 1, repeat: int = 5, **kwargs) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    cost_time = (time.time() - start_time) * 1000 / repeat  # unit: ms
    return cost_time


def set_random_seed(seed: int) -> None:
    import random

    random.seed(seed)
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def post_empty_cache(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return result

    return wrapper
