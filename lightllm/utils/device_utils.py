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


@lru_cache(maxsize=None)
def init_p2p(device_index):
    """
    torch 调用跨卡的to操作后，triton编译的算子便能自动操作跨卡tensor。
    """
    import torch

    num_gpus = torch.cuda.device_count()
    tensor = torch.zeros((1,))
    tensor = tensor.to(f"cuda:{device_index}")
    for j in range(num_gpus):
        tensor.to(f"cuda:{j}")

    torch.cuda.empty_cache()
    return


@lru_cache(maxsize=None)
def kv_trans_use_p2p():
    return os.getenv("KV_TRANS_USE_P2P", "False").upper() in ["1", "TRUE", "ON"]
