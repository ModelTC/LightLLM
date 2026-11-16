import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from validate_glm53_optim import validate


def validate_kernels(gpu):
    result = validate(gpu)
    import flash_kda
    import sgl_deep_gemm
    from lightllm.common.basemodel.triton_kernel.fused_moe.sm90_fp8_mega_moe import get_sm90_mega_moe_module
    from lightllm.server.api_cli import make_argument_parser

    assert importlib.metadata.version("flash-kda") == "0.0.1+d7e3b3a9"
    assert callable(flash_kda.fwd)
    assert get_sm90_mega_moe_module() is sgl_deep_gemm
    for name in ("fp8_mega_moe", "mega_moe_pre_dispatch_sm90", "get_symm_buffer_for_sm90_mega_moe"):
        assert callable(getattr(sgl_deep_gemm, name)), name
    receipt = json.loads(Path("/opt/lightllm-build/kernel_build.json").read_text())
    native = Path(sgl_deep_gemm.__file__).parent / "_C.so"
    assert hashlib.sha256(native.read_bytes()).hexdigest() == receipt["sgl_deep_gemm_native_sha256"]
    args = make_argument_parser().parse_args(
        ["--llm_prefill_att_backend", "auto", "flashkda", "--ep_moe_backend", "megamoe", "--expert_dtype", "fp8"]
    )
    assert args.llm_prefill_att_backend == ["auto", "flashkda"] and args.ep_moe_backend == "megamoe"
    result["kernels"] = receipt
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_kernels(args.gpu), indent=2))
