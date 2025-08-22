import os
import time
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
import multiprocessing.shared_memory as shm
from lightllm.utils.envs_utils import get_env_start_args


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


def create_afs(name, data):
    try:
        data_size = len(data)
        path = os.path.join(get_env_start_args().visual_embed_path, name)
        with open(path, "xb") as f:
            mem_view = memoryview(data)
            f.write(mem_view[:data_size])
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        print("Warning create afs {} failed because of FileExistsError!".format(name))


def read_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def read_afs(name: str, base_dir: str = "/mtc/sangchengmeng/afs") -> bytes:

    path = Path(base_dir) / name
    return path.read_bytes()


def free_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def free_afs(name):
    path = os.path.join(get_env_start_args().visual_embed_path, name)
    try:
        os.remove(path)
    except FileNotFoundError:
        print("Warning free afs {} failed because of FileNotFoundError!".format(name))
        return
    except PermissionError as e:
        print("Warning free afs {} failed due to PermissionError: {}".format(name, e))
        return


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"


def afs_embed_exists(md5sum: str):
    uid_int = int(md5sum, 16)
    filename = f"{uid_int}-embed"
    fullpath = os.path.join(get_env_start_args().visual_embed_path, filename)
    return True if os.path.isfile(fullpath) else False
