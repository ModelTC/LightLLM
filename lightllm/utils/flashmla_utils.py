"""Resolve the native FlashMLA implementation without requiring duplicate wheels."""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_flash_mla():
    try:
        import flash_mla
    except ModuleNotFoundError as exc:
        if exc.name != "flash_mla":
            raise
        # SGL Kernel also packages the native Hopper V4 sparse decode extension
        # and exposes the same metadata / KV-cache API. Numerical and graph
        # regressions must validate the selected implementation before deployment.
        from sgl_kernel import flash_mla
    return flash_mla
