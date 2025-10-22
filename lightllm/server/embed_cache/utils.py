import torch
import numpy as np
from io import BytesIO
from typing import List, Optional
import multiprocessing.shared_memory as shm


def tensor2bytes(t: torch.Tensor):
    # t = t.cpu().numpy().tobytes()
    # return t
    buf = BytesIO()
    t = t.detach().cpu()
    # 这个地方进行新的empty并复制是因为，torch的tensor save的机制存在问题
    # 如果 t 是从一个大 tensor 上切片复制下来的的tensor， 在save的时候，其
    # 会保存大tensor的所有数据，所以会导致存储开销较大，需要申请一个新的tensor
    # 并进行复制，来打断这种联系。
    dest = torch.empty_like(t)
    dest.copy_(t)
    torch.save(dest, buf, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buf.seek(0)
    return buf.read()


def list2bytes(tensors: List[torch.Tensor]) -> bytes:
    # 逐个张量做 detach().cpu() 和复制
    safe_list = []
    for t in tensors:
        if t is None:
            safe_list.append(None)
            continue
        t = t.detach().cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        dest = torch.empty_like(t)
        dest.copy_(t)
        safe_list.append(dest)
    buf = BytesIO()
    torch.save(safe_list, buf, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buf.seek(0)
    return buf.read()


def bytes2tensor(b):
    # return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    return torch.load(BytesIO(b), weights_only=False)


def bytes2list(b: bytes, device: Optional[torch.device] = None, non_blocking: bool = False) -> List[torch.Tensor]:
    obj = torch.load(BytesIO(b), map_location="cpu", weights_only=False)

    if isinstance(obj, tuple):
        obj = list(obj)
    if not isinstance(obj, list):
        raise TypeError(f"Loaded object is {type(obj)}, expected list or tuple.")

    if device is None:
        return obj

    out: List[torch.Tensor] = []
    for x in obj:
        if x is None:
            out.append(None)
        elif isinstance(x, torch.Tensor):
            out.append(x.to(device, non_blocking=non_blocking))
        else:
            raise TypeError(f"List element is {type(x)}, expected Tensor or None.")
    return out


def create_shm(name, data):
    try:
        data_size = len(data)
        shared_memory = shm.SharedMemory(name=name, create=True, size=data_size)
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def read_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def free_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"


def get_shm_name_deepstack(uid):
    return str(uid) + "-deepstack"
