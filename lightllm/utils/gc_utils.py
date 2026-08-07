import gc
from contextlib import contextmanager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def freeze_gc(tag: str = "") -> None:
    gc.collect()
    gc.freeze()
    logger.info(f"gc.freeze done ({tag}): frozen={gc.get_freeze_count()}")


@contextmanager
def gc_frozen_and_disabled(tag: str = ""):
    freeze_gc(tag)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
