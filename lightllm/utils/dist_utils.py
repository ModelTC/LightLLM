import torch.distributed as dist


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        raise RuntimeError("Distributed package is not initialized.")


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        raise RuntimeError("Distributed package is not initialized.")
