"""Validate the overlay's packages and control-plane APIs without model requests.

--gpu additionally checks native GPU imports on a Hopper host. It does not run a
model or claim to validate multi-node communication or numerical kernel results.
"""

import argparse
import importlib
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def validate(gpu=False):
    root = Path(__file__).resolve().parents[1]
    versions = {}
    pins = {}
    for filename in ("requirements.glm53_optim.txt", "requirements.glm53_optim_extra.txt"):
        for line in (root / "docker" / filename).read_text().splitlines():
            if not line or line.startswith(("#", "--")):
                continue
            name, expected = line.split(" --hash=", 1)[0].split("==")
            assert name not in pins or pins[name] == expected, f"Conflicting pin: {name}"
            pins[name] = expected
    for name, expected in pins.items():
        actual = importlib.metadata.version(name)
        assert actual == expected, f"{name}: expected {expected}, found {actual}"
        versions[name] = actual
        distribution = importlib.metadata.distribution(name)
        missing = [
            str(item)
            for item in distribution.files or ()
            if not str(item).endswith(".pyc") and not distribution.locate_file(item).exists()
        ]
        assert not missing, f"{name}: installed metadata lists missing files: {missing[:8]}"

    import torch
    import triton
    import lightllm
    import deep_gemm
    import deep_ep
    import light_mem
    import tilelang
    import flashinfer
    import xgrammar as xgr
    from torch_memory_saver import torch_memory_saver, configure_subprocess
    from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t, nixlBind
    from lightllm.utils.flashmla_utils import get_flash_mla

    assert torch.__version__ == "2.11.0+cu130", torch.__version__
    assert triton.__version__ == "3.6.0", triton.__version__
    assert Path(lightllm.__file__).resolve().parent == root / "lightllm"
    for name in ("get_num_sms", "set_num_sms", "fp8_gemm_nt", "m_grouped_fp8_gemm_nt_masked"):
        assert callable(getattr(deep_gemm, name)), name
    assert callable(deep_ep.Buffer)
    receipt = json.loads(Path("/opt/lightllm-build/deepep_rcqp.json").read_text())
    native = Path(deep_ep.__file__).parent / Path(receipt["native_path"]).name
    assert hashlib.sha256(native.read_bytes()).hexdigest() == receipt["native_sha256"]
    for name, field in (("nvidia-nvshmem-cu13", "nvshmem"), ("nvidia-nccl-cu13", "nccl")):
        assert importlib.metadata.version(name) == receipt[field]
    assert callable(light_mem.PyLocalCacheService)
    assert hasattr(light_mem.PyState, "Finished")
    assert callable(torch_memory_saver.region) and callable(configure_subprocess)
    assert callable(nixl_agent)
    nixl_agent_config(sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW)
    for name in ("nixlCancelledError", "nixlRemoteDisconnectError", "nixlNotPostedError"):
        assert issubclass(getattr(nixlBind, name), Exception), name

    flash_mla = get_flash_mla()
    assert callable(flash_mla.flash_mla_with_kvcache)
    assert flash_mla.get_mla_metadata()[0] is not None
    info = xgr.TokenizerInfo([bytes([i]) for i in range(256)], vocab_type=xgr.VocabType.RAW)
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    grammar = compiler.compile_json_schema(
        '{"type":"object","properties":{"ok":{"type":"boolean"}},' '"required":["ok"],"additionalProperties":false}'
    )
    assert xgr.GrammarMatcher(grammar).accept_string('{"ok":true}')
    assert not xgr.GrammarMatcher(grammar).accept_string('{"ok":5}')

    for module in (
        "cutlass",
        "cutlass.cute",
        "cuda.tile",
        "cudnn",
        "nccl.core",
        "lightllm.models.glm5_next.model",
        "lightllm.models.glm5_next_mtp.model",
        "lightllm.common.basemodel.triton_kernel.linear_att.replayssm",
        "lightllm.common.basemodel.attention.nsa.fp8_glm5_next",
        "lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter",
        "lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware",
    ):
        importlib.import_module(module)

    cli = subprocess.run(
        [sys.executable, "-m", "lightllm.server.api_server", "--help"],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    for option in ("--ssm_state_mode", "--ssm_replay_config", "--config_server_port", "--llm_kv_type"):
        assert option in cli.stdout, option

    native_gpu_imports = "not run; use --gpu on a Hopper host with a CUDA 13 compatible driver"
    if gpu:
        assert torch.cuda.is_available(), "CUDA is unavailable"
        assert torch.cuda.get_device_capability() == (9, 0), "Target is Hopper (SM90)"
        import sgl_kernel
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        assert callable(flash_mla_sparse_fwd) and callable(flash_attn_with_kvcache)
        native_gpu_imports = {
            "device": torch.cuda.get_device_name(),
            "sgl_kernel": sgl_kernel.__file__,
            "kernel_numerics": "not tested by this preflight",
        }

    revision = root / ".git"
    commit = (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision.exists()
        else None
    )
    stamped = Path("/opt/lightllm-build/source_revision")
    if stamped.exists():
        assert stamped.read_text().strip() == commit, "Source revision differs from image stamp"
    return {
        "status": "passed",
        "source": str(root),
        "commit": commit,
        "versions": versions,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "tilelang": tilelang.__version__,
        "flashinfer": flashinfer.__version__,
        "native_gpu_imports": native_gpu_imports,
        "deepep_rcqp": receipt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(gpu=args.gpu), ensure_ascii=False, indent=2))
