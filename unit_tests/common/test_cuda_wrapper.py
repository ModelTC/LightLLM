"""Optional batch-copy bindings must not change the legacy CUDA wrapper contract."""

import io

import pytest

from lightllm.common import cuda_wrapper


def test_legacy_wrapper_does_not_require_cuda13_batch_symbols(monkeypatch):
    requested = []

    class Function:
        def __call__(self, *args):
            return 0

    class LegacyLibrary:
        def __getattr__(self, name):
            assert name not in ("cudaMemcpyBatchAsync", "cudaRuntimeGetVersion")
            requested.append(name)
            return Function()

    monkeypatch.setattr(cuda_wrapper.ctypes, "CDLL", lambda path: LegacyLibrary())
    monkeypatch.setattr(cuda_wrapper.CudaRTLibrary, "path_to_library_cache", {})
    monkeypatch.setattr(cuda_wrapper.CudaRTLibrary, "path_to_dict_mapping", {})
    wrapper = cuda_wrapper.CudaRTLibrary("legacy-cudart")
    wrapper.cudaDeviceSynchronize()
    assert set(requested) == {function.name for function in wrapper.exported_functions}


@pytest.mark.parametrize(
    ("maps", "major_version", "expected"),
    [
        ("7f /cuda-12/libcudart.so.12.8\n7f /cuda-13/libcudart.so.13\n", None, "/cuda-12/libcudart.so.12.8"),
        ("7f /cuda-12/libcudart.so.12.8\n7f /cuda-13/libcudart.so.13\n", 13, "/cuda-13/libcudart.so.13"),
        ("7f /cuda-13/libcudart-hash.so.13.0 (deleted)\n", 13, "/cuda-13/libcudart-hash.so.13.0"),
        ("7f /libcudart/other.so.13\n7f /cuda/libcudart.so.13\n", 13, "/cuda/libcudart.so.13"),
        ("7f /cuda/libcudart.so.130\n7f /cuda/libcudart.so.14\n", 13, None),
        ("7f libcudart.so.13\n7f /cuda/libcudart.so.13\n", 13, "/cuda/libcudart.so.13"),
        ("7f /cuda/libcudart.so\n", None, "/cuda/libcudart.so"),
        ("", None, None),
    ],
)
def test_find_loaded_library_filters_major_version(monkeypatch, maps, major_version, expected):
    monkeypatch.setattr("builtins.open", lambda _path: io.StringIO(maps))
    assert cuda_wrapper.find_loaded_library("libcudart", major_version=major_version) == expected


def test_find_loaded_library_returns_none_when_maps_unavailable(monkeypatch):
    def unavailable(_path):
        raise OSError("maps unavailable")

    monkeypatch.setattr("builtins.open", unavailable)
    assert cuda_wrapper.find_loaded_library("libcudart") is None
    assert cuda_wrapper.find_loaded_library("libcudart", major_version=13) is None
