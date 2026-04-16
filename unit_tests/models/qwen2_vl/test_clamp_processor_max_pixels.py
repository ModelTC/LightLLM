import importlib.util
import logging
import os
import re
import unittest


# Load just the clamp helper's source so we don't drag torch/triton/HF into
# pytest collection just to test a pure function. Stubbing modules would leak
# into other tests collected in the same session.
_VP_PATH = os.path.normpath(
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
with open(_VP_PATH, "r") as _f:
    _src = _f.read()
_match = re.search(r"^def clamp_processor_max_pixels\b.*?(?=^def |\Z)", _src, re.DOTALL | re.MULTILINE)
assert _match, "clamp_processor_max_pixels not found in vision_process.py"
_ns = {"logger": logging.getLogger("clamp_test")}
exec("from typing import Optional\n" + _match.group(0), _ns)
clamp_processor_max_pixels = _ns["clamp_processor_max_pixels"]


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

    def test_none_max_pixels_gets_populated(self):
        # HF-loaded processors can have max_pixels=None; treat as "unset, populate".
        p = _FakeProcessor(patch_size=14, merge_size=2, max_pixels=None)
        clamp_processor_max_pixels(p, visual_image_max_tokens=4096)
        self.assertEqual(p.max_pixels, 4096 * 28 * 28)

    def test_missing_patch_or_merge_size_is_noop(self):
        # A non-Qwen-VL processor may lack these attributes entirely.
        p = _FakeProcessor(patch_size=None, merge_size=2, max_pixels=1000)
        clamp_processor_max_pixels(p, visual_image_max_tokens=4096)
        self.assertEqual(p.max_pixels, 1000)

        p2 = _FakeProcessor(patch_size=14, merge_size=None, max_pixels=1000)
        clamp_processor_max_pixels(p2, visual_image_max_tokens=4096)
        self.assertEqual(p2.max_pixels, 1000)


if __name__ == "__main__":
    unittest.main()
