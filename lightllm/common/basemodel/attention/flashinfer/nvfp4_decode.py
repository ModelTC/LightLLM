"""Select tile-local NVFP4 staging for Hopper's small-query FA2 decode kernel.

The generated module gets its own launcher and cache identity. Neither installed
FlashInfer headers nor global module generators are modified.
"""

import inspect
from types import SimpleNamespace

from ...flashinfer_kernel.nvfp4_decode import get_nvfp4_decode_module


# The FA2 FFI entry point's argument order. Resolve the Python wrapper's fields by
# name instead of relying on offsets for optional parameters added by FlashInfer.
_FFI_FIELDS = (
    "float_workspace_buffer",
    "int_workspace_buffer",
    "plan_info_vec",
    "q",
    "paged_k_cache",
    "paged_v_cache",
    "qo_indptr",
    "paged_kv_indptr",
    "paged_kv_indices",
    "paged_kv_last_page_len",
    "o",
    "maybe_lse",
    "mask_mode",
    "layout",
    "window_left",
    "enable_pdl",
    "maybe_custom_mask",
    "maybe_mask_indptr",
    "maybe_alibi_slopes",
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "key_block_scales",
    "value_block_scales",
    "logits_soft_cap",
    "sm_scale",
    "rope_scale",
    "rope_theta",
    "token_pos_in_items_len",
)


def use_small_tile_nvfp4_kernel(wrapper, dtype, head_dim):
    """Replace this decode wrapper's module while preserving its graph metadata."""
    module = get_nvfp4_decode_module(dtype, head_dim)
    if module is None:
        return
    native_module = wrapper._cached_module
    names = tuple(inspect.signature(native_module.paged_run).parameters)
    indices = tuple(names.index(name) for name in _FFI_FIELDS)

    def paged_run(*args):
        values = [args[index] for index in indices]
        # The Python wrapper takes RoPE scales; the FFI takes their reciprocals.
        values[-3] = 1.0 / values[-3]
        values[-2] = 1.0 / values[-2]
        return module.paged_run(*values)

    wrapper._cached_module = SimpleNamespace(
        plan=module.plan, paged_run=paged_run, workspace_size=getattr(module, "workspace_size", None)
    )
