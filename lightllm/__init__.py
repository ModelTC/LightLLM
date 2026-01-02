from lightllm.utils.device_utils import is_musa

if is_musa():
    import torchada  # noqa: F401
