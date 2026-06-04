import os
import ctypes
import torch
from typing import Any, Tuple
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.platform import get_backend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def _cuda_worker(shm_ptr: int, tasks: Tuple[int ,int], handle: Any):
    cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
    cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cuda.cudaHostRegister.restype = ctypes.c_int
    cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    cuda.cudaHostGetDevicePointer.restype = ctypes.c_int

    cudaHostRegisterFlag = 3

    torch.cuda.set_device(get_current_device_id())
    # TODO 这个地方的分块注册是否具备合法性和合理性。
    for offset, seg_len in tasks:
        ptr = ctypes.c_void_p(shm_ptr + offset)
        r = cuda.cudaHostRegister(ptr, ctypes.c_size_t(seg_len), cudaHostRegisterFlag)
        if r != 0:
            raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")
        handle.task_count += 1

        if handle.device_ptr is None:
            # 提前获取对应的指针对象，避免在wait后再获取，照成过长的阻塞等待。
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"cudaHostGetDevicePointer failed with error code {res}")

            logger.info(
                f"cudaHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}"
            )
            handle.device_ptr = device_ptr.value

    handle.tasks_finished.set()


def _npu_worker(shm_ptr: int, tasks: Tuple[int ,int], handle: Any):
    import acl

    acl.init()
    ret = acl.rt.set_device(get_current_device_id())
    assert ret == 0, f"acl.rt.set_device failed with error code {ret}"

    ACL_HOST_REGISTER_MAPPED = 0
    for offset, seg_len in tasks:
        ptr = shm_ptr + offset
        res = acl.rt.host_register(ptr, seg_len, ACL_HOST_REGISTER_MAPPED)
        assert res[1] == 0, f"acl.rt.host_register failed with error code {res}"
        handle.task_count += 1

    handle.tasks_finished.set()


def _metax_worker(shm_ptr: int, tasks: Tuple[int ,int], handle: Any):
    mc = ctypes.CDLL(os.path.join(os.getenv("MACA_PATH", "/opt/maca"), "lib/libmcruntime.so"))
    mc.mcHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    mc.mcHostRegister.restype = ctypes.c_int
    mc.mcHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
    mc.mcHostGetDevicePointer.restype = ctypes.c_int

    cudaHostRegisterFlag = 3

    torch.cuda.set_device(get_current_device_id())
    # TODO 这个地方的分块注册是否具备合法性和合理性。
    for offset, seg_len in tasks:
        ptr = ctypes.c_void_p(shm_ptr + offset)
        r = mc.mcHostRegister(ptr, ctypes.c_size_t(seg_len), cudaHostRegisterFlag)
        if r != 0:
            raise Exception(f"cudaHostRegister failed with error code {r}, prefer to use hugetlb")
        handle.task_count += 1

        if handle.device_ptr is None:
            # 提前获取对应的指针对象，避免在wait后再获取，照成过长的阻塞等待。
            device_ptr = ctypes.c_void_p()
            host_ptr = ctypes.c_void_p(shm_ptr)
            res = mc.mcHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
            if res != 0:
                raise Exception(f"mcHostGetDevicePointer failed with error code {res}")

            logger.info(
                f"mcHostGetDevicePointer success, host_ptr={host_ptr.value}, device_ptr={device_ptr.value}"
            )
            handle.device_ptr = device_ptr.value

    handle.tasks_finished.set()


def get_host_register_worker():
    backend_name = get_backend().name
    if backend_name == "cuda":
        return _cuda_worker
    elif backend_name == "ascend":
        return _npu_worker
    elif backend_name == "maca":
        return _metax_worker
    else:
        raise RuntimeError(f"platform {backend_name} is not registered!")

