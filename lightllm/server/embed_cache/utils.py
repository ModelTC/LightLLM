import torch
import numpy as np
from io import BytesIO
import multiprocessing.shared_memory as shm


def tensor2bytes(t):
    t = t.cpu().numpy().tobytes()
    return t
    # buf = BytesIO()
    # torch.save(t, buf)
    # buf.seek(0)
    # return buf.read()


def bytes2tensor(b):
    return torch.from_numpy(np.frombuffer(b, dtype=np.float16)).cuda()
    # return torch.load(BytesIO(b))


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
