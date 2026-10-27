import os

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# Features that depend on cuda-only libraries or CUDA IPC and have no xpu implementation yet.
_UNSUPPORTED_FLAGS = {
    "enable_cpu_cache": "the cpu kv cache registers host memory with cudaHostRegister",
    "enable_disk_cache": "the disk cache is built on the cpu kv cache",
    "enable_dp_prompt_cache_fetch": "it shares kv buffers across processes with CUDA IPC",
    "enable_ep_moe": "expert parallel MoE needs DeepEP V2 and DeepGEMM",
    "enable_torch_memory_saver": "torch_memory_saver is cuda only",
    "enable_prefill_cudagraph": "xpu graph capture is not enabled yet",
    "enable_mps": "MPS is an NVIDIA feature",
}


def check_and_set_xpu_args(args) -> None:
    """Reject start args that cannot run on xpu and default to eager mode."""
    for flag, reason in _UNSUPPORTED_FLAGS.items():
        if getattr(args, flag, False):
            raise ValueError(f"--{flag} is not supported with --hardware_platform xpu: {reason}.")

    if args.run_mode in ["prefill", "decode"]:
        raise ValueError(
            "PD disaggregation (--run_mode prefill/decode) is not supported with --hardware_platform xpu: "
            "the kv transfer uses CUDA IPC and NCCL."
        )
    if args.mtp_mode is not None:
        raise ValueError("--mtp_mode is not supported with --hardware_platform xpu yet, it needs graph capture.")
    if args.sampling_backend == "flashinfer":
        raise ValueError("--sampling_backend flashinfer is not supported with --hardware_platform xpu, use triton.")
    if args.enable_profiling == "nvtx":
        raise ValueError("--enable_profiling nvtx is not supported with --hardware_platform xpu.")

    # xpu graph capture is not validated yet, run eager unless explicitly opted in.
    if not args.disable_cudagraph and os.getenv("LIGHTLLM_XPU_ENABLE_GRAPH", "0").upper() not in ["1", "TRUE", "ON"]:
        logger.info("graph capture is disabled on xpu, set LIGHTLLM_XPU_ENABLE_GRAPH=1 to try it.")
        args.disable_cudagraph = True
