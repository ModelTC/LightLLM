from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

FLASHINFER_AVAILABLE = False
flashinfer = None
try:
    import flashinfer as _flashinfer

    flashinfer = _flashinfer
    FLASHINFER_AVAILABLE = False
except ImportError:
    logger.warning("flashinfer is not available")
