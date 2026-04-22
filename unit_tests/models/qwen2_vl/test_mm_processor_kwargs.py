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
        with open(_PATH, "r") as f:
            src = f.read()
        start = src.index("def apply_mm_processor_kwargs")
        tail = src[start:]
        next_def = tail.find("\ndef ", 1)
        fn_src = tail[:next_def] if next_def != -1 else tail
        ns = {}
        import logging

        ns["logger"] = logging.getLogger("mm_kwargs_test")
        exec("from typing import Any, Dict, Optional\n" + fn_src, ns)
        return ns["apply_mm_processor_kwargs"]
    return mod.apply_mm_processor_kwargs


apply_mm_processor_kwargs = _load_helper()


class _FakeProcessor:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class TestApplyMMProcessorKwargs(unittest.TestCase):
    def test_none_is_noop(self):
        p = _FakeProcessor(max_pixels=16384 * 28 * 28, min_pixels=4 * 28 * 28)
        apply_mm_processor_kwargs(p, None)
        self.assertEqual(p.max_pixels, 16384 * 28 * 28)
        self.assertEqual(p.min_pixels, 4 * 28 * 28)

    def test_empty_dict_is_noop(self):
        p = _FakeProcessor(max_pixels=16384 * 28 * 28)
        apply_mm_processor_kwargs(p, {})
        self.assertEqual(p.max_pixels, 16384 * 28 * 28)

    def test_sets_existing_attribute(self):
        p = _FakeProcessor(max_pixels=16384 * 28 * 28)
        apply_mm_processor_kwargs(p, {"max_pixels": 1024 * 28 * 28})
        self.assertEqual(p.max_pixels, 1024 * 28 * 28)

    def test_adds_new_attribute(self):
        p = _FakeProcessor(max_pixels=100)
        apply_mm_processor_kwargs(p, {"some_new_key": 42})
        self.assertEqual(p.some_new_key, 42)

    def test_multiple_keys(self):
        p = _FakeProcessor(max_pixels=16384 * 28 * 28, min_pixels=4 * 28 * 28)
        apply_mm_processor_kwargs(p, {"max_pixels": 1000 * 28 * 28, "min_pixels": 16 * 28 * 28})
        self.assertEqual(p.max_pixels, 1000 * 28 * 28)
        self.assertEqual(p.min_pixels, 16 * 28 * 28)

    def test_overrides_none_attribute(self):
        p = _FakeProcessor(max_pixels=None)
        apply_mm_processor_kwargs(p, {"max_pixels": 2000 * 28 * 28})
        self.assertEqual(p.max_pixels, 2000 * 28 * 28)

    def test_max_pixels_mirrors_into_size_longest_edge(self):
        # HF Qwen3-VL processor exposes pixel bounds via the ``size`` dict
        # (shortest_edge / longest_edge). The tokenizer reads from there for
        # get_image_token_length, so setting .max_pixels alone is not enough.
        p = _FakeProcessor(
            max_pixels=12845056,
            min_pixels=3136,
            size={"shortest_edge": 3136, "longest_edge": 12845056},
        )
        apply_mm_processor_kwargs(p, {"max_pixels": 1003520, "min_pixels": 3136})
        self.assertEqual(p.max_pixels, 1003520)
        self.assertEqual(p.size["longest_edge"], 1003520)
        self.assertEqual(p.size["shortest_edge"], 3136)

    def test_size_dict_unchanged_for_unrelated_keys(self):
        p = _FakeProcessor(
            max_pixels=1,
            size={"shortest_edge": 3136, "longest_edge": 12845056},
        )
        apply_mm_processor_kwargs(p, {"some_other_key": 7})
        self.assertEqual(p.size, {"shortest_edge": 3136, "longest_edge": 12845056})

    def test_missing_size_dict_is_ok(self):
        # Not all processors expose a ``size`` dict; mirroring must be a no-op
        # rather than crashing when it's absent.
        p = _FakeProcessor(max_pixels=1)
        apply_mm_processor_kwargs(p, {"max_pixels": 1003520})
        self.assertEqual(p.max_pixels, 1003520)
        self.assertFalse(hasattr(p, "size"))


if __name__ == "__main__":
    unittest.main()
