import torch


def custom_cat(tensors):
    """
    直接调用 torch 的 cat操作，会造成多个流同步阻塞，用 custom_cat 进行替换。
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError("Input must be a list of tensors")

    assert tensors[0].is_cuda and len(tensors[0].shape) == 1
    sizes = [t.shape[0] for t in tensors]
    dest_size = sum(sizes)
    out_tensor = torch.empty((dest_size,), dtype=tensors[0].dtype, device=tensors[0].device)

    start_loc = 0
    for t, size in zip(tensors, sizes):
        out_tensor[start_loc : (start_loc + size)].copy_(t, non_blocking=True)
        start_loc += size

    return out_tensor
