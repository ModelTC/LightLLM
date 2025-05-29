from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    # TODO: lightllm_kernel release
    import lightllm_kernel

    light_ops = getattr(lightllm_kernel, "ops", lightllm_kernel)
    HAS_LIGHTLLM_KERNEL = True
except:
    light_ops = None
    HAS_LIGHTLLM_KERNEL = False
    logger.warning("lightllm_kernel is not installed, you can't use the api of it.")
