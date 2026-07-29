from typing import Optional


QUANT_TYPE_NONE = "none"
QUANT_TYPE_AWQ = "awq"
QUANT_TYPE_AWQ_MARLIN = "awq_marlin"
QUANT_TYPE_W8A8_VLLM = "w8a8-vllm"
QUANT_TYPE_FP8W8A8_VLLM = "fp8w8a8-vllm"
QUANT_TYPE_FP8W8A8_PT_VLLM = "fp8w8a8-pt-vllm"
QUANT_TYPE_FP8W8A8_PT_SGL = "fp8w8a8-pt-sgl"
QUANT_TYPE_FP8W8A8_PT_TRITON = "fp8w8a8-pt-triton"
QUANT_TYPE_FP8W8A8_B128_VLLM = "fp8w8a8-b128-vllm"
QUANT_TYPE_FP8W8A8_B128_DEEPGEMM = "fp8w8a8-b128-deepgemm"
QUANT_TYPE_FP8W8A8_B128_TRITON = "fp8w8a8-b128-triton"
QUANT_TYPE_FP8W8A8G128_TRITON = "fp8w8a8g128-triton"
QUANT_TYPE_FP8W8A8G64_TRITON = "fp8w8a8g64-triton"
QUANT_TYPE_FP4FP8_B32_DEEPGEMM = "fp4fp8-b32-deepgemm"


# Public CLI names and their internal canonical names. Short names select the
# default backend, while explicit names always end with the backend.
QUANT_TYPE_CANONICAL_MAP = {
    "w8a8": QUANT_TYPE_W8A8_VLLM,
    "fp8w8a8": QUANT_TYPE_FP8W8A8_VLLM,
    "fp8w8a8-pt": QUANT_TYPE_FP8W8A8_PT_TRITON,
    "fp8w8a8-b128": QUANT_TYPE_FP8W8A8_B128_TRITON,
    "fp8w8a8g128": QUANT_TYPE_FP8W8A8G128_TRITON,
    "fp8w8a8g64": QUANT_TYPE_FP8W8A8G64_TRITON,
    QUANT_TYPE_AWQ: QUANT_TYPE_AWQ,
    QUANT_TYPE_AWQ_MARLIN: QUANT_TYPE_AWQ_MARLIN,
    QUANT_TYPE_NONE: QUANT_TYPE_NONE,
    QUANT_TYPE_W8A8_VLLM: QUANT_TYPE_W8A8_VLLM,
    QUANT_TYPE_FP8W8A8_VLLM: QUANT_TYPE_FP8W8A8_VLLM,
    QUANT_TYPE_FP8W8A8_PT_VLLM: QUANT_TYPE_FP8W8A8_PT_VLLM,
    QUANT_TYPE_FP8W8A8_PT_SGL: QUANT_TYPE_FP8W8A8_PT_SGL,
    QUANT_TYPE_FP8W8A8_PT_TRITON: QUANT_TYPE_FP8W8A8_PT_TRITON,
    QUANT_TYPE_FP8W8A8_B128_VLLM: QUANT_TYPE_FP8W8A8_B128_VLLM,
    QUANT_TYPE_FP8W8A8_B128_DEEPGEMM: QUANT_TYPE_FP8W8A8_B128_DEEPGEMM,
    QUANT_TYPE_FP8W8A8_B128_TRITON: QUANT_TYPE_FP8W8A8_B128_TRITON,
    QUANT_TYPE_FP8W8A8G128_TRITON: QUANT_TYPE_FP8W8A8G128_TRITON,
    QUANT_TYPE_FP8W8A8G64_TRITON: QUANT_TYPE_FP8W8A8G64_TRITON,
    QUANT_TYPE_FP4FP8_B32_DEEPGEMM: QUANT_TYPE_FP4FP8_B32_DEEPGEMM,
}

SUPPORTED_QUANT_TYPES = tuple(QUANT_TYPE_CANONICAL_MAP)
CANONICAL_QUANT_TYPES = frozenset(QUANT_TYPE_CANONICAL_MAP.values())
SUPPORTED_VIT_QUANT_TYPES = (
    "w8a8",
    "fp8w8a8",
    QUANT_TYPE_NONE,
    QUANT_TYPE_W8A8_VLLM,
    QUANT_TYPE_FP8W8A8_VLLM,
)
CANONICAL_VIT_QUANT_TYPES = frozenset(
    {
        QUANT_TYPE_NONE,
        QUANT_TYPE_W8A8_VLLM,
        QUANT_TYPE_FP8W8A8_VLLM,
    }
)


def normalize_quant_type(quant_type: Optional[str]) -> str:
    if quant_type is None:
        return QUANT_TYPE_NONE
    try:
        return QUANT_TYPE_CANONICAL_MAP[quant_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported quant_type `{quant_type}`; expected one of {list(SUPPORTED_QUANT_TYPES)}"
        ) from exc


def normalize_vit_quant_type(quant_type: Optional[str]) -> str:
    canonical_name = normalize_quant_type(quant_type)
    if canonical_name not in CANONICAL_VIT_QUANT_TYPES:
        raise ValueError(
            f"unsupported vit_quant_type `{quant_type}`; expected one of {list(SUPPORTED_VIT_QUANT_TYPES)}"
        )
    return canonical_name
