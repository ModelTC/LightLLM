from lightllm.platform.base.registry import register_platform

register_platform(
    "cuda",
    backend="lightllm.platform.backends.cuda_like:CudaBackend",
)

register_platform(
    "musa",
    backend="lightllm.platform.backends.cuda_like:MusaBackend",
)

register_platform(
    "ascend",
    backend="lightllm.platform.backends.ascend:AscendBackend",
)
