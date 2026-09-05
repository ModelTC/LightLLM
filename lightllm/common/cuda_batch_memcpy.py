"""Lazy CUDA Runtime wrapper for batched asynchronous copies.

``cudaMemcpyBatchAsync`` was added before CUDA 13, but CUDA 13 removed the
``failIdx`` parameter from its ABI.  LightLLM requires the CUDA Runtime 13.x
ABI for same-node EPLB source-push; unsupported runtimes fail fast.
"""

import ctypes
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


class CudaBatchMemcpyUnavailable(RuntimeError):
    """The current process cannot safely use CUDA batched memcpy."""


class cudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class cudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", cudaMemLocation),
        ("dstLocHint", cudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


# cudaMemcpySrcAccessOrderStream and cudaMemcpyFlagPreferOverlapWithCompute.
_SRC_ACCESS_ORDER_STREAM = 1
_PREFER_OVERLAP_WITH_COMPUTE = 1
_CUDA_13_0 = 13000
_CUDA_14_0 = 14000


def _find_loaded_cudart() -> Optional[str]:
    """Return libcudart's mapped path without loading CUDA as a side effect."""

    paths = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libcudart" not in line:
                    continue
                path_start = line.find("/")
                if path_start >= 0:
                    path = line[path_start:].strip().removesuffix(" (deleted)")
                    if path not in paths:
                        paths.append(path)
    except OSError:
        pass
    if not paths:
        return None

    # PyTorch can map both its CUDA 12 compatibility runtime and CUDA 13's
    # runtime.  Select CUDA 13 explicitly instead of guessing a future ABI.
    def runtime_major(path):
        match = re.search(r"\.so\.(\d+)(?:\D|$)", path)
        return int(match.group(1)) if match else 0

    cuda_13_paths = [path for path in paths if runtime_major(path) == 13]
    return cuda_13_paths[0] if cuda_13_paths else None


@dataclass
class PreparedCudaMemcpyBatch:
    """Host-side arrays retained for one cudaMemcpyBatchAsync submission."""

    dsts: object
    srcs: object
    sizes: object
    attrs: cudaMemcpyAttributes
    attrs_idxs: object
    count: int


class CudaBatchMemcpy:
    """CUDA 13.x ``cudaMemcpyBatchAsync`` binding.

    This class does not load libcudart when CUDA is absent: construction first
    finds the runtime already loaded by PyTorch.
    """

    def __init__(self, library=None):
        if library is None:
            path = _find_loaded_cudart()
            if path is None:
                raise CudaBatchMemcpyUnavailable("libcudart.so.13 is not loaded")
            try:
                library = ctypes.CDLL(path)
            except OSError as exc:
                raise CudaBatchMemcpyUnavailable(f"cannot load libcudart: {exc}") from exc

        try:
            self._runtime_get_version = library.cudaRuntimeGetVersion
            self._batch_async = library.cudaMemcpyBatchAsync
            self._get_error_string = library.cudaGetErrorString
        except AttributeError as exc:
            raise CudaBatchMemcpyUnavailable("cudaMemcpyBatchAsync is unavailable") from exc

        self._runtime_get_version.restype = ctypes.c_int
        self._runtime_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._get_error_string.restype = ctypes.c_char_p
        self._get_error_string.argtypes = [ctypes.c_int]
        runtime_version = ctypes.c_int()
        result = self._runtime_get_version(ctypes.byref(runtime_version))
        if result != 0:
            raise CudaBatchMemcpyUnavailable(f"cudaRuntimeGetVersion failed with CUDA error {result}")
        if not _CUDA_13_0 <= runtime_version.value < _CUDA_14_0:
            raise CudaBatchMemcpyUnavailable(
                f"cudaMemcpyBatchAsync requires CUDA Runtime 13.x (13.0 ABI), found {runtime_version.value}"
            )
        self.runtime_version = runtime_version.value

        pointer_array = ctypes.POINTER(ctypes.c_void_p)
        self._batch_async.restype = ctypes.c_int
        # CUDA 13 ABI: dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream.
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
    def prepare(copies: Iterable[Tuple[int, int, int]]) -> PreparedCudaMemcpyBatch:
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
        attrs.srcAccessOrder = _SRC_ACCESS_ORDER_STREAM
        # srcLocHint and dstLocHint are intentionally zero/unspecified.
        attrs.flags = _PREFER_OVERLAP_WITH_COMPUTE
        attrs_idxs = (ctypes.c_size_t * 1)(0)
        return PreparedCudaMemcpyBatch(dsts, srcs, sizes, attrs, attrs_idxs, count)

    def enqueue(self, prepared: PreparedCudaMemcpyBatch, stream: int) -> None:
        if prepared.count <= 0:
            raise ValueError("cudaMemcpyBatchAsync requires at least one copy")
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
