import os
from functools import lru_cache


def set_current_device_id(device_id: int):
    os.environ["CURRENT_DEVICE_ID"] = str(device_id)


@lru_cache(maxsize=None)
def get_current_device_id():
    import torch

    if torch.cuda.is_available():
        device_id = os.getenv("CURRENT_DEVICE_ID", None)
        if device_id is None:
            raise RuntimeError("set_current_device_id must called first to set current device")
        return int(device_id)
    else:
        raise RuntimeError("Torch CUDA is not avaliable.")


@lru_cache(maxsize=None)
def get_device_sm_count():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["multiprocessor_count"]


@lru_cache(maxsize=None)
def get_device_sm_regs_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_num_regs"]


@lru_cache(maxsize=None)
def get_device_sm_shared_mem_num():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["max_shared_mem"]


@lru_cache(maxsize=None)
def get_device_warp_size():
    import triton
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(0)
    return properties["warpSize"]


@lru_cache(maxsize=None)
def get_current_device_name():
    import torch

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        return gpu_name
    else:
        return None
