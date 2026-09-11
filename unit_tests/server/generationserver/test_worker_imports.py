import subprocess
import sys
from pathlib import Path


def test_action_model_worker_imports_in_clean_interpreter():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lightllm.server.generationserver.action.model_rpc",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
