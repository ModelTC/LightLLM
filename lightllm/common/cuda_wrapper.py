# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/distributed/device_communicators/cuda_wrapper.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python cudart wrapper with asynchronous pinned-source lifetime management."""

import ctypes
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class PinnedH2DBatchCopier:
    """每个推理进程共享批量 H2D 状态，统一管理异步源保活与异常回退。"""

    _native_batch_available = True
    # 原生 batch 禁止 legacy 默认流，仅在默认流调用时按设备延迟创建上传流。
    _upload_streams: Dict[torch.device, torch.cuda.Stream] = {}
    # 只在原生 H2D 成功提交、需要后台保活时创建线程池，导入和 copy_ 回退均不创建。
    _release_pool: Optional[ThreadPoolExecutor] = None
    _release_pool_lock = Lock()

    @classmethod
    def copy_async(cls, dst_gpu_tensors: List[torch.Tensor], src_pinned_tensors: List[torch.Tensor]):
        """按当前流顺序批量异步 H2D，并保护 pinned 源直到传输结束。

        源由 PyTorch pinned allocator 申请，各对 Tensor 连续且形状和 dtype 一致。
        默认流的原生 batch 适配封装在内部；调用方只遵循当前流顺序，不阻塞 CPU。
        上传完成前不能主动改写源，目标须保持有效到传输结束。
        """
        if not dst_gpu_tensors:
            return
        stream = torch.cuda.current_stream()
        if cls._native_batch_available and stream.cuda_stream == 0:
            # 原生 API 不接受 legacy 默认流；上传流先等待当前流，提交后再交回依赖。
            upload_stream = cls._get_upload_stream(stream.device)
            upload_stream.wait_stream(stream)
            with torch.cuda.stream(upload_stream):
                cls._copy_on_stream(dst_gpu_tensors, src_pinned_tensors, upload_stream)
            stream.wait_stream(upload_stream)
        else:
            cls._copy_on_stream(dst_gpu_tensors, src_pinned_tensors, stream)

    @classmethod
    def _get_upload_stream(cls, device: torch.device) -> torch.cuda.Stream:
        if device not in cls._upload_streams:
            cls._upload_streams[device] = torch.cuda.Stream(device=device)
        return cls._upload_streams[device]

    @classmethod
    def _copy_on_stream(
        cls, dst_gpu_tensors: List[torch.Tensor], src_pinned_tensors: List[torch.Tensor], stream: torch.cuda.Stream
    ):
        """在已选定的当前流提交，沿用原生保活和 PyTorch 回退。"""
        if cls._native_batch_available and cls._try_native_batch_h2d(dst_gpu_tensors, src_pinned_tensors, stream):
            cls._protect_pinned_sources(src_pinned_tensors, stream)
            return

        # PyTorch copy_ 已经登记 host allocator，回退路径无需额外保活或登记。
        for dst, src in zip(dst_gpu_tensors, src_pinned_tensors):
            dst.copy_(src, non_blocking=True)

    @classmethod
    def _try_native_batch_h2d(
        cls, dst_gpu_tensors: List[torch.Tensor], src_pinned_tensors: List[torch.Tensor], stream: torch.cuda.Stream
    ) -> bool:
        """尝试原生批量提交；接口准备或提交失败时禁用该路径，让调用方回退到 copy_。"""
        try:
            from cuda.bindings import runtime as cuda_runtime

            attrs = cuda_runtime.cudaMemcpyAttributes()
            # Stream 模式让源读取遵循上传流顺序；提交成功后由完成事件和后台任务保活。
            attrs.srcAccessOrder = cuda_runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
            dst_ptrs = [tensor.data_ptr() for tensor in dst_gpu_tensors]
            src_ptrs = [tensor.data_ptr() for tensor in src_pinned_tensors]
            byte_sizes = [tensor.numel() * tensor.element_size() for tensor in src_pinned_tensors]
            batch_copy = cuda_runtime.cudaMemcpyBatchAsync
        except Exception as error:
            cls._native_batch_available = False
            logger.warning(
                "Native batch H2D API unavailable; "
                f"disabling native batch copy and falling back to PyTorch copy_: {error}"
            )
            return False

        try:
            result = batch_copy(dst_ptrs, src_ptrs, byte_sizes, len(byte_sizes), [attrs], [0], 1, stream.cuda_stream)
            error = result[0]
            if error != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyBatchAsync failed: {error.name}")
        except Exception as error:
            cls._native_batch_available = False
            logger.warning(
                "Native batch H2D submission failed; "
                f"disabling native batch copy and falling back to PyTorch copy_: {error}"
            )
            return False
        return True

    @classmethod
    def _protect_pinned_sources(cls, src_pinned_tensors: List[torch.Tensor], stream: torch.cuda.Stream):
        """每批 raw H2D 在上传流记录一个完成事件，由复用线程池接管源 Tensor 引用。"""
        # blocking=True 让后台线程休眠等待；Event.synchronize 会释放 GIL，不阻塞主推理线程。
        # 事件排在 raw batch 之后，覆盖此前所有源读取；调用方的 wait_stream 负责 GPU 数据依赖。
        event = torch.cuda.Event(blocking=True)
        event.record(stream)
        # 必须复制容器：调用方可能立即清空自己的列表，切片本身仍持有整块 pinned storage。
        cls._get_release_pool().submit(cls._wait_and_release_pinned_sources, event, list(src_pinned_tensors))

    @classmethod
    def _get_release_pool(cls) -> ThreadPoolExecutor:
        if cls._release_pool is None:
            # 首次创建加锁，避免多个提交线程同时建立等待线程池；后续调用直接复用。
            with cls._release_pool_lock:
                if cls._release_pool is None:
                    cls._release_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lightllm-pinned-release")
        return cls._release_pool

    @classmethod
    def _wait_and_release_pinned_sources(cls, event: torch.cuda.Event, sources: List[torch.Tensor]):
        """只管理 CPU 源的生命周期，不修改请求状态，也不向流提交新的 GPU 工作。"""
        event.synchronize()
        # 普通工作线程可安全释放 Tensor；不受 CUDA host callback 禁止调用 CUDA API 的限制。
        # 不把 sources 返回给 Future，避免任务结束后仍被结果对象持有。
        sources.clear()


def cuda_memcpy_pinned_h2d_batch_async(dst_gpu_tensors: List[torch.Tensor], src_pinned_tensors: List[torch.Tensor]):
    """按当前流顺序批量异步 H2D，默认流适配、源内存保活和回退由封装统一管理。"""
    PinnedH2DBatchCopier.copy_async(dst_gpu_tensors, src_pinned_tensors)


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class cudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class cudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", cudaMemLocation),
        ("dstLocHint", cudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


@dataclass
class PreparedMemcpyBatch:
    """Host-side arrays retained for one cudaMemcpyBatchAsync submission."""

    dsts: object
    srcs: object
    sizes: object
    attrs: cudaMemcpyAttributes
    attrs_idxs: object
    count: int


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name: str, *, major_version: Optional[int] = None) -> Optional[str]:
    """
    According to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. Return the first matching library path,
    optionally filtering by the shared library's major version.
    """  # noqa
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                if lib_name not in line:
                    continue
                path_start = line.find("/")
                if path_start < 0:
                    continue
                # Also accept hashed filenames such as libcudart-hash.so.13.0.
                path = line[path_start:].strip().removesuffix(" (deleted)")
                filename = path.rsplit("/", 1)[-1]
                if not filename.rpartition(".so")[0].startswith(lib_name):
                    continue
                if major_version is not None and not re.search(rf"\.so\.{major_version}(?:\D|$)", filename):
                    continue
                return path
    except OSError:
        pass
    return None


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t 	cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char* 	cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t 	cudaMalloc ( void** devPtr, size_t size )
        Function("cudaMalloc", cudaError_t, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function("cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function("cudaMemcpy", cudaError_t, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind]),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function("cudaIpcGetMemHandle", cudaError_t, [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p]),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle", cudaError_t, [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint]
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            assert so_file is not None, "libcudart is not loaded in the current process"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr))
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess)
        )
        return devPtr


class CudaBatchMemcpy:
    """CUDA 13.x ``cudaMemcpyBatchAsync`` binding for batched memory copies."""

    _SRC_ACCESS_ORDER_STREAM = 1
    _PREFER_OVERLAP_WITH_COMPUTE = 1
    _CUDA_13_0 = 13000
    _CUDA_14_0 = 14000

    def __init__(self, library=None):
        if library is None:
            path = find_loaded_library("libcudart", major_version=13)
            if path is None:
                raise RuntimeError(
                    "CUDA batch copies require CUDA Runtime 13.x cudaMemcpyBatchAsync; " "libcudart.so.13 is not loaded"
                )
            try:
                library = ctypes.CDLL(path)
            except OSError as exc:
                raise RuntimeError(f"cannot load libcudart: {exc}") from exc

        try:
            runtime_get_version = library.cudaRuntimeGetVersion
            self._batch_async = library.cudaMemcpyBatchAsync
            self._get_error_string = library.cudaGetErrorString
        except AttributeError as exc:
            raise RuntimeError("cudaMemcpyBatchAsync is unavailable") from exc

        runtime_get_version.restype = ctypes.c_int
        runtime_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._get_error_string.restype = ctypes.c_char_p
        self._get_error_string.argtypes = [ctypes.c_int]
        runtime_version = ctypes.c_int()
        result = runtime_get_version(ctypes.byref(runtime_version))
        if result != 0:
            raise RuntimeError(f"cudaRuntimeGetVersion failed with CUDA error {result}")
        if not self._CUDA_13_0 <= runtime_version.value < self._CUDA_14_0:
            raise RuntimeError(
                f"cudaMemcpyBatchAsync requires CUDA Runtime 13.x (13.0 ABI), found {runtime_version.value}"
            )

        pointer_array = ctypes.POINTER(ctypes.c_void_p)
        self._batch_async.restype = ctypes.c_int
        self._batch_async.argtypes = [
            pointer_array,
            pointer_array,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(cudaMemcpyAttributes),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

    @staticmethod
    def prepare_memcpy_batch(copies: Iterable[Tuple[int, int, int]]) -> PreparedMemcpyBatch:
        copies = tuple(copies)
        if not copies:
            raise ValueError("cudaMemcpyBatchAsync requires at least one copy")
        for src, dst, size in copies:
            if not src or not dst or size <= 0:
                raise ValueError("cudaMemcpyBatchAsync requires non-null pointers and positive sizes")
        count = len(copies)
        dsts = (ctypes.c_void_p * count)(*(dst for _, dst, _ in copies))
        srcs = (ctypes.c_void_p * count)(*(src for src, _, _ in copies))
        sizes = (ctypes.c_size_t * count)(*(size for _, _, size in copies))
        attrs = cudaMemcpyAttributes()
        attrs.srcAccessOrder = CudaBatchMemcpy._SRC_ACCESS_ORDER_STREAM
        attrs.flags = CudaBatchMemcpy._PREFER_OVERLAP_WITH_COMPUTE
        attrs_idxs = (ctypes.c_size_t * 1)(0)
        return PreparedMemcpyBatch(dsts, srcs, sizes, attrs, attrs_idxs, count)

    def cudaMemcpyBatchAsync(self, prepared: PreparedMemcpyBatch, stream: int) -> None:
        result = self._batch_async(
            prepared.dsts,
            prepared.srcs,
            prepared.sizes,
            prepared.count,
            ctypes.byref(prepared.attrs),
            prepared.attrs_idxs,
            1,
            ctypes.c_void_p(stream),
        )
        if result != 0:
            message = self._get_error_string(result)
            error = message.decode("utf-8") if message else f"CUDA error {result}"
            raise RuntimeError(f"cudaMemcpyBatchAsync failed: {error}")
