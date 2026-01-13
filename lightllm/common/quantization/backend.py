import os
from enum import Enum, auto
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class BackendType(Enum):
    TRITON = auto()
    VLLM = auto()
    DEEPGEMM = auto()


class BackendRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._force_triton = os.getenv("LIGHTLLM_USE_TRITON_QUANT", "0").upper() in ["1", "TRUE", "ON"]

        self._has_vllm = self._check_vllm()
        self._has_deepgemm = self._check_deepgemm()

        if self._force_triton:
            logger.info("LIGHTLLM_USE_TRITON_QUANT is set, forcing Triton backend for quantization")
        else:
            logger.info(f"Available quantization backends: vLLM={self._has_vllm}, DeepGEMM={self._has_deepgemm}")

    def _check_vllm(self) -> bool:
        try:
            from lightllm.utils.vllm_utils import HAS_VLLM

            return HAS_VLLM
        except ImportError:
            return False

    def _check_deepgemm(self) -> bool:
        try:
            import deep_gemm  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def force_triton(self) -> bool:
        return self._force_triton

    @property
    def has_vllm(self) -> bool:
        return self._has_vllm

    @property
    def has_deepgemm(self) -> bool:
        return self._has_deepgemm

    def get_backend(self, quant_type: str) -> BackendType:
        if self._force_triton:
            return BackendType.TRITON

        if quant_type == "fp8-block128":
            if self._has_deepgemm:
                return BackendType.DEEPGEMM
            elif self._has_vllm:
                return BackendType.VLLM
        elif quant_type in ["w8a8", "fp8-per-token"]:
            if self._has_vllm:
                return BackendType.VLLM

        return BackendType.TRITON


QUANT_BACKEND = BackendRegistry()
