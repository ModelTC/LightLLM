"""Backend validators for attention backends.

This module provides validation for attention backends (FA3, FlashInfer, Triton)
by running actual operations and comparing against ground truth.
"""

import multiprocessing as mp
from lightllm.utils.log_utils import init_logger

from .base import BackendValidator
from .fa3 import FA3Validator
from .flashinfer import FlashInferValidator
from .triton import TritonValidator

logger = init_logger(__name__)

# Timeout for subprocess validation in seconds
_VALIDATION_TIMEOUT = 30

# Registry of backend validators
VALIDATORS = {
    "fa3": FA3Validator,
    "flashinfer": FlashInferValidator,
    "triton": TritonValidator,
}


def _run_validation_in_subprocess(backend_name: str, result_pipe) -> None:
    """Execute validation in a subprocess.

    This isolates validation failures (including crashes) from the main process.
    """
    try:
        validator = VALIDATORS.get(backend_name)
        if validator is None:
            result_pipe.send((False, f"Unknown backend: {backend_name}"))
            return

        success, error = validator.validate()
        result_pipe.send((success, error))

    except Exception as e:
        result_pipe.send((False, str(e)))


def validate_backend(backend_name: str) -> bool:
    """Validate a backend by running it in a subprocess.

    Args:
        backend_name: One of "fa3", "flashinfer", "triton"

    Returns:
        True if validation passed, False otherwise.
    """
    validator = VALIDATORS.get(backend_name)
    if validator is None:
        logger.warning(f"Unknown backend: {backend_name}")
        return False

    # Quick availability check before spawning subprocess
    if not validator.is_available():
        logger.info(f"Backend {backend_name} dependencies not available")
        return False

    try:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_run_validation_in_subprocess, args=(backend_name, child_conn))
        proc.start()
        proc.join(timeout=_VALIDATION_TIMEOUT)

        if proc.is_alive():
            proc.kill()
            proc.join()
            logger.warning(f"Backend {backend_name} validation timed out")
            return False

        if proc.exitcode != 0:
            logger.warning(f"Backend {backend_name} validation subprocess crashed (exit code {proc.exitcode})")
            return False

        if parent_conn.poll():
            success, error = parent_conn.recv()
            if success:
                logger.info(f"Backend {backend_name} validated successfully")
                return True
            else:
                logger.warning(f"Backend {backend_name} validation failed: {error}")
                return False
        else:
            logger.warning(f"Backend {backend_name} validation produced no result")
            return False

    except Exception as e:
        logger.warning(f"Backend {backend_name} validation exception: {e}")
        return False


def is_backend_available(backend_name: str) -> bool:
    """Check if a backend's dependencies are available (quick check, no subprocess)."""
    validator = VALIDATORS.get(backend_name)
    if validator is None:
        return False
    return validator.is_available()


__all__ = [
    "BackendValidator",
    "FA3Validator",
    "FlashInferValidator",
    "TritonValidator",
    "validate_backend",
    "is_backend_available",
    "VALIDATORS",
]
