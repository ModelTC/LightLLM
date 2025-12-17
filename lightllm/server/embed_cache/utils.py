import torch
import numpy as np
from io import BytesIO
import multiprocessing.shared_memory as shm
import time


def tensor2bytes(t: torch.Tensor):
    # 转换为 numpy array，使用 contiguous 确保内存连续
    print(f"tensor2bytes shape: {t.shape} {t.is_contiguous()}")
    memory_size = t.numel() * t.element_size()
    out = torch.empty(memory_size, dtype=torch.uint8, device="cpu", pin_memory=True).copy_(t.view(torch.uint8).view(-1))
    return out.numpy().tobytes()


def bytes2tensor(b):
    # 直接返回二进制数据的 uint8 tensor，外部自己转 dtype 和 view
    # 避免 numpy 不支持 bfloat16 等问题
    return torch.frombuffer(b, dtype=torch.uint8)


def create_shm_and_dump(name, data: torch.Tensor):
    try:
        data_size = data.numel() * data.element_size()
        shared_memory = shm.SharedMemory(name=name, create=True, size=data_size)
        tensor = torch.frombuffer(shared_memory.buf, dtype=torch.uint8)
        out = torch.empty(data_size, dtype=torch.uint8, device="cpu", pin_memory=True).copy_(
            data.view(torch.uint8).view(-1)
        )
        tensor.copy_(out)
        return tensor
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


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
