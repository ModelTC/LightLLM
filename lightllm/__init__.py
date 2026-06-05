from lightllm.utils.device_utils import is_musa


def _patch_mp_resource_tracker_for_semaphore():
    from multiprocessing import resource_tracker

    if getattr(resource_tracker, "_lightllm_ignore_semaphore", False):
        return

    orig_register = resource_tracker.register
    orig_unregister = resource_tracker.unregister

    def register(name, rtype):
        if rtype == "semaphore":
            return
        return orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "semaphore":
            return
        return orig_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister
    resource_tracker._lightllm_ignore_semaphore = True


_patch_mp_resource_tracker_for_semaphore()

if is_musa():
    import torchada  # noqa: F401
