"""Root conftest.py for LightLLM test suite.

Patches ``transformers.utils.versions`` before any test module is imported so
that a tokenizers version mismatch (e.g. tokenizers 0.22.x vs the <0.22
requirement baked into transformers 4.49) doesn't prevent ``api_models`` from
being imported in the unit-test environment.

This is a test-environment shim only; production code never calls this.
"""
import sys
from unittest.mock import MagicMock

# Only install the shim if transformers hasn't been imported yet AND it would
# fail the version check. We do this unconditionally here so that any test
# file that imports ``lightllm.server.api_models`` can do so safely.
if "transformers.utils.versions" not in sys.modules:
    _mock_versions = MagicMock()
    _mock_versions.require_version = lambda *a, **kw: None
    _mock_versions.require_version_core = lambda *a, **kw: None
    sys.modules["transformers.utils.versions"] = _mock_versions
