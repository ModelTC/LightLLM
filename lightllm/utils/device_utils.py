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
