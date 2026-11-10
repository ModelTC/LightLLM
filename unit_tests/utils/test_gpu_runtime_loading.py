"""Check CUDA/HIP library discovery without requiring those installation layouts."""

import ctypes
import unittest
from unittest.mock import patch

from lightllm.utils import kv_cache_utils


class TestGpuRuntimeLoading(unittest.TestCase):
    def load_from(self, available, hip=None):
        def load(name):
            if name not in available:
                raise OSError("library unavailable in this installation")
            return name

        with patch.object(kv_cache_utils.torch.version, "hip", hip):
            with patch.object(kv_cache_utils.ctypes, "CDLL", side_effect=load):
                return kv_cache_utils._load_gpu_runtime()

    def test_cuda_and_hip_layouts(self):
        cases = [
            (None, "libcudart.so", "cuda"),
            (None, "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so", "cuda"),
            (None, "/usr/local/cuda/lib64/libcudart.so", "cuda"),
            (None, "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so", "cuda"),
            ("6.0", "libamdhip64.so", "hip"),
            ("6.0", "/opt/rocm/lib/libamdhip64.so", "hip"),
        ]
        for hip, library, prefix in cases:
            with self.subTest(library=library):
                self.assertEqual(self.load_from({library}, hip), (library, prefix))

    def test_preserves_legacy_x86_precedence(self):
        legacy = "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so"
        generic = "/usr/local/cuda/lib64/libcudart.so"
        self.assertEqual(self.load_from({legacy, generic}), (legacy, "cuda"))

    def test_missing_runtime_reports_failure(self):
        for hip in (None, "6.0"):
            with self.subTest(hip=hip):
                with self.assertRaisesRegex(OSError, "cannot load GPU runtime"):
                    self.load_from(set(), hip)

    def test_glibc_exports_shared_memory_functions(self):
        libc = ctypes.CDLL("libc.so.6")
        for name in ("shmget", "shmat", "shmctl"):
            self.assertTrue(callable(getattr(libc, name)))


if __name__ == "__main__":
    unittest.main()
