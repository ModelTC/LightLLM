import os
import subprocess
import sys


def test_api_and_selection_do_not_import_provider_extensions():
    code = """
import sys
import lightllm.common.attention.api
import lightllm.common.attention.selection
assert not any(name.startswith(('flashinfer', 'flash_mla', 'flash_qla')) for name in sys.modules)
assert not any(name.startswith('lightllm.common.attention.backends.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
