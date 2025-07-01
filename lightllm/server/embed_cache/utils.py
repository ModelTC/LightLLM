import torch
import numpy as np
from io import BytesIO
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


def bytes2tensor(b):
    # return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    return torch.load(BytesIO(b), weights_only=False)


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
