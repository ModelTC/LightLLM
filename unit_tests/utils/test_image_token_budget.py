import importlib.util
import os
import unittest

# Load the helper directly to avoid triggering heavy package imports (torch,
# atomics, etc.) that the full lightllm package pulls in.
_UTILS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "lightllm",
        "utils",
        "multimodal_utils.py",
    )
)
_spec = importlib.util.spec_from_file_location("_mm_utils_under_test", _UTILS_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
enforce_image_token_budget = _module.enforce_image_token_budget


class TestEnforceImageTokenBudget(unittest.TestCase):
    def test_none_budget_allows_anything(self):
        enforce_image_token_budget(token_num=10_000_000, max_tokens=None)

    def test_under_budget_ok(self):
        enforce_image_token_budget(token_num=1000, max_tokens=1024)

    def test_at_budget_ok(self):
        enforce_image_token_budget(token_num=1024, max_tokens=1024)

    def test_over_budget_raises(self):
        with self.assertRaises(ValueError) as cm:
            enforce_image_token_budget(token_num=2048, max_tokens=1024, image_index=3)
        msg = str(cm.exception)
        self.assertIn("image[3]", msg)
        self.assertIn("2048", msg)
        self.assertIn("1024", msg)

    def test_zero_budget_rejects_any_positive_tokens(self):
        with self.assertRaises(ValueError):
            enforce_image_token_budget(token_num=1, max_tokens=0)


if __name__ == "__main__":
    unittest.main()
