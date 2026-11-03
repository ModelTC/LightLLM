from functools import partial, singledispatch

from .copy_plan import BoundedPageCopy, bind_window_page_copy
from .sliding_window.fp8_e8m0_swa_state import FP8E8M0SlidingWindowState
from .sliding_window.fp8_e8m0_swa_transfer import copy_swa_window
from .window_compressor.window_compressor_state import WindowCompressorState
from .window_compressor.window_compressor_transfer import copy_compressor_window


@singledispatch
def bind_runtime_page_copy(state, pages, layout, write):
    return BoundedPageCopy.bind(state.get_runtime_transfer_view(), pages, layout, write).run_pd


@bind_runtime_page_copy.register(FP8E8M0SlidingWindowState)
def _bind_sliding_window(state, pages, layout, write):
    return bind_window_page_copy(
        state.get_runtime_transfer_view(), pages, layout, write, partial(copy_swa_window, config=state.config)
    )


@bind_runtime_page_copy.register(WindowCompressorState)
def _bind_compressor(state, pages, layout, write):
    return bind_window_page_copy(state.get_runtime_transfer_view(), pages, layout, write, copy_compressor_window)
