import importlib.util
import os
import unittest

# Load the helper directly from its file so we do not trigger heavy imports in
# lightllm.models.* (torch, triton kernels, etc.) just to test a pure function.
_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "lightllm",
        "models",
        "qwen2_vl",
        "vision_process.py",
    )
)


def _load_helper():
    import sys
    import types

    # Stub out heavy imports that vision_process.py pulls at module load.
    # Only the pure helper is under test; nothing below depends on these stubs.
    for name in ("torch", "numpy", "PIL", "PIL.Image"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv_t = types.ModuleType("torchvision.transforms")
        tv_tv2 = types.ModuleType("torchvision.transforms.v2")
        tv_tf = types.ModuleType("torchvision.transforms.v2.functional")
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = tv_t
        sys.modules["torchvision.transforms.v2"] = tv_tv2
        sys.modules["torchvision.transforms.v2.functional"] = tv_tf

    # The file imports transformers pieces; stub them.
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = types.ModuleType("transformers")
    for sub in (
        "transformers.image_utils",
        "transformers.image_processing_utils_fast",
        "transformers.image_transforms",
    ):
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)

    spec = importlib.util.spec_from_file_location("_vp_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # If stubs aren't enough to import the whole file, fall back to
        # reading the function source and exec'ing it directly.
        with open(_PATH, "r") as f:
            src = f.read()
        start = src.index("def clamp_processor_max_pixels")
        # Find the end — the next "def " at column 0.
        tail = src[start:]
        next_def = tail.find("\ndef ", 1)
        fn_src = tail[:next_def] if next_def != -1 else tail
        ns = {}
        # Substitute logger with a noop.
        import logging

        ns["logger"] = logging.getLogger("clamp_test")
        exec("from typing import Optional\n" + fn_src, ns)
        return ns["clamp_processor_max_pixels"]
    return mod.clamp_processor_max_pixels


clamp_processor_max_pixels = _load_helper()


class _FakeProcessor:
    def __init__(self, patch_size, merge_size, max_pixels):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_pixels = max_pixels


class TestClampProcessorMaxPixels(unittest.TestCase):
    def test_none_budget_is_noop(self):
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=16384 * 28 * 28)
        clamp_processor_max_pixels(p, None)
        self.assertEqual(p.max_pixels, 16384 * 28 * 28)

    def test_budget_looser_than_processor_is_noop(self):
        # Processor's max_pixels already gives 16384 tokens. Budget is 32768. Keep smaller.
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=16384 * 28 * 28)
        clamp_processor_max_pixels(p, visual_image_max_tokens=32768)
        self.assertEqual(p.max_pixels, 16384 * 28 * 28)

    def test_budget_tighter_clamps(self):
        # patch=14, merge=2 -> unit=28, unit^2=784. Budget 4096 tokens -> 4096*784 pixels.
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=16384 * 28 * 28)
        clamp_processor_max_pixels(p, visual_image_max_tokens=4096)
        self.assertEqual(p.max_pixels, 4096 * 28 * 28)

    def test_budget_equal_to_original_is_noop(self):
        # Original gives exactly 16384 tokens. Budget 16384 -> same value.
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=16384 * 28 * 28)
        clamp_processor_max_pixels(p, visual_image_max_tokens=16384)
        self.assertEqual(p.max_pixels, 16384 * 28 * 28)

    def test_budget_zero_raises(self):
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=16384 * 28 * 28)
        with self.assertRaises(ValueError):
            clamp_processor_max_pixels(p, visual_image_max_tokens=0)

    def test_different_patch_merge(self):
        # patch=16, merge=1 -> unit=16, unit^2=256. Budget 1000 tokens -> 256000 pixels.
        p = _FakeProcessor(patch_size=16, merge_size=1, max_pixels=10_000_000)
        clamp_processor_max_pixels(p, visual_image_max_tokens=1000)
        self.assertEqual(p.max_pixels, 1000 * 16 * 16)

    def test_processor_max_pixels_none_is_clamped(self):
        # HF Qwen3.5-VL's processor exposes max_pixels=None (no intrinsic upper
        # bound); the clamp must treat that as "looser than any budget" and
        # always apply our allowed_max_pixels instead of crashing on int<None.
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=None)
        clamp_processor_max_pixels(p, visual_image_max_tokens=4096)
        self.assertEqual(p.max_pixels, 4096 * 28 * 28)


if __name__ == "__main__":
    unittest.main()
