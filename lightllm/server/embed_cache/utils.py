from lightllm.utils.shm_utils import ServiceSharedMemory, get_service_shm_name


def create_shm(name, data):
    name = get_service_shm_name(name)
    try:
        data_size = len(data)
        shared_memory = ServiceSharedMemory(name=name, create=True, size=data_size)
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def read_shm(name):
    name = get_service_shm_name(name)
    shared_memory = ServiceSharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def free_shm(name):
    name = get_service_shm_name(name)
    shared_memory = ServiceSharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def get_shm_name_data(uid):
    return f"{uid}-data"
