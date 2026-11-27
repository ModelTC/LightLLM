"""Probe for the Neo build of FA3 (flash_attn_interface), whose kernel takes image_token_end.

LightLLM's regular FA3 comes from sgl_kernel; Neo models' bidirectional image-span prefill needs
this separate build.
"""

import inspect
import warnings

NEO_FA3_INSTALL_HINT = (
    "Install the Neo FA3 build from source in the current Python environment:\n"
    "  git clone --recursive https://github.com/WANDY666/flash-attention.git\n"
    "  cd flash-attention/hopper\n"
    "  FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install "
    "--no-build-isolation --no-deps --force-reinstall .\n"
    "This replaces a standalone flash_attn_3 / flash_attn_interface installation; "
    "LightLLM's regular FA3 uses the separate sgl_kernel package."
)


try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_neo

    # Neo FA3 accepts the exclusive visible KV end for each image query.
    _sig = inspect.signature(flash_attn_with_kvcache_neo)
    if "image_token_end" not in _sig.parameters:
        raise ImportError("flash_attn_interface is missing image_token_end support (need the Neo build)")

    HAS_FLASH_ATTN_INTERFACE = True
except ImportError as exc:
    warnings.warn(
        f"Neo FA3 is unavailable: {exc}. "
        "Automatic Neo prefill selection will fall back to Triton. "
        "To select Triton explicitly, use --att_backend gqa.prefill=triton.\n" + NEO_FA3_INSTALL_HINT
    )
    flash_attn_with_kvcache_neo = None
    HAS_FLASH_ATTN_INTERFACE = False
