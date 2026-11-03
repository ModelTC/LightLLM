"""Process-local ownership of CUDA host registrations (not of the backing storage)."""

import ctypes
import concurrent.futures
import os
import time
import weakref

import torch

from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
_registrations = weakref.WeakSet()


class HostMemoryRegistration:
    """Keep every successful registration base until it has been unregistered.

    The caller owns the mapping and must keep it alive, stop all GPU consumers,
    and synchronize their streams before close(). Registration failure rolls back
    only after all registration threads have finished.
    """

    def __init__(self, host_ptr: int, size: int, chunk_bytes: int, workers: int):
        self.host_ptr = host_ptr
        self.size = size
        self.device_id = get_current_device_id()
        torch.cuda.set_device(self.device_id)
        self.cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")
        self.cuda.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuda.cudaHostRegister.restype = ctypes.c_int
        self.cuda.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuda.cudaHostUnregister.restype = ctypes.c_int
        self.cuda.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
        self.cuda.cudaHostGetDevicePointer.restype = ctypes.c_int
        self.bases = []
        _registrations.add(self)
        self._finalizer = weakref.finalize(self, self._unregister, self.cuda, self.bases, self.device_id)
        started = time.monotonic()

        def register(offset):
            torch.cuda.set_device(self.device_id)
            base = host_ptr + offset
            result = self.cuda.cudaHostRegister(
                ctypes.c_void_p(base), ctypes.c_size_t(min(chunk_bytes, size - offset)), 3
            )
            if result != 0:
                raise RuntimeError(f"cudaHostRegister({base}) failed: {result}")
            return base

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(register, offset) for offset in range(0, size, chunk_bytes)]
                errors = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        self.bases.append(future.result())
                    except Exception as error:
                        errors.append(error)
            if errors:
                raise errors[0]
            device_ptr = ctypes.c_void_p()
            result = self.cuda.cudaHostGetDevicePointer(ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), 0)
            if result != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer failed: {result}")
            # All consumers use one contiguous UVA view across independent segments.
            if device_ptr.value != host_ptr:
                raise RuntimeError("CUDA host cache requires identical host and device UVA addresses")
            self.device_ptr = device_ptr.value
        except BaseException:
            self.close()
            raise
        logger.info(
            f"pid {os.getpid()} registered {size} host bytes in {len(self.bases)} segments "
            f"with {workers} threads: {time.monotonic() - started:.3f}s"
        )

    @staticmethod
    def _unregister(cuda, bases, device_id, progress=None):
        if not bases:
            return
        torch.cuda.set_device(device_id)
        errors = []
        for base in list(bases):
            result = cuda.cudaHostUnregister(ctypes.c_void_p(base))
            if result == 0:
                bases.remove(base)
                if progress is not None:
                    progress()
            else:
                errors.append((base, result))
        if errors:
            raise RuntimeError(f"cudaHostUnregister failed: {errors}")

    def close(self, progress=None):
        """Idempotent; failed ranges remain owned so a later close can retry."""
        if not self.bases:
            return
        started = time.monotonic()
        logger.info(f"pid {os.getpid()} unregistering {self.size} host bytes in {len(self.bases)} segments")
        self._unregister(self.cuda, self.bases, self.device_id, progress)
        logger.info(f"pid {os.getpid()} unregistered {self.size} host bytes in {time.monotonic() - started:.3f}s")


def close_host_memory_registrations(progress=None):
    """Call only after this process has stopped and synchronized its GPU consumers."""
    errors = []
    for registration in list(_registrations):
        try:
            registration.close(progress=progress)
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(f"Failed to release {len(errors)} host registrations") from errors[0]
