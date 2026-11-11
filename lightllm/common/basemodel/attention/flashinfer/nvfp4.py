"""Native paged NVFP4 attention, with graph metadata shared with FlashInfer."""

import inspect
import weakref

import torch

from .fp import FlashInferAttBackend, FlashInferDecodeAttState, FlashInferPrefillAttState
from ..base_att import AttControl
from ..triton.nvfp4 import NVFP4TritonAttBackend


def nvfp4_paged_views(k, v, page_size):
    # Storage appends E4M3 scale bytes to each head's packed E2M1 payload.
    # FlashInfer accepts their strides directly: no repacking or cache copy.
    width = k.shape[-1]
    dim = width * 16 // 9
    kp = k.view(-1, page_size, k.shape[1], width)
    vp = v.view(-1, page_size, v.shape[1], width)
    return (
        (kp[..., : dim // 2], vp[..., : dim // 2]),
        (kp[..., dim // 2 :].view(torch.float8_e4m3fn), vp[..., dim // 2 :].view(torch.float8_e4m3fn)),
    )


class NVFP4FlashInferAttBackend(FlashInferAttBackend):
    supports_parallel_block_decode = True
    workspace_buffer_key = "flashinfer_nvfp4"
    _initialized = False

    def __init__(self, model):
        # BaseAttBackend caches one instance per model, but Python calls
        # __init__ again when FA3 requests its over-budget paged fallback.
        if self._initialized:
            return
        self.model = model
        self.fallback = NVFP4TritonAttBackend(model)
        dim = model.config.get("head_dim", model.config["hidden_size"] // model.config["num_attention_heads"])
        try:
            import flashinfer

            # The interleaved scale row must preserve FA2's 8-byte alignment.
            # D64 has a 36-byte row; use the stride-safe Triton path there.
            self.native_supported = (
                dim in (128, 256)
                and "kv_cache_sf" in inspect.signature(flashinfer.BatchDecodeWithPagedKVCacheWrapper.run).parameters
                and torch.cuda.get_device_capability() != (10, 7)
            )
        except ImportError:
            self.native_supported = False
        if self.native_supported:
            super().__init__(model)
            self.kv_data_type = torch.uint8
            # Replicated KV layouts can have more local heads than global/TP.
            # Plan against the actual registered storage, as the FA3 staging
            # path does. Heterogeneous head layouts keep the shape-aware fallback.
            manager = getattr(model, "att_state_manager", None)
            if manager is not None:
                shapes = {
                    (state.config.head_num, state.config.head_dim)
                    for state in manager.get_full_attention_states(role=model.att_state_role)
                }
                if len(shapes) == 1:
                    self.tp_kv_head_num, stored_dim = shapes.pop()
                    self.native_supported = stored_dim == self.head_dim
                elif shapes:
                    self.native_supported = False
        self._initialized = True

    def _needs_fallback(self):
        model = self.model
        return (
            not self.native_supported
            or not self.uses_causal_attention()
            or model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model) != 1
        )

    def create_att_prefill_state(self, infer_state):
        if self._needs_fallback():
            return self.fallback.create_att_prefill_state(infer_state)
        return NVFP4FlashInferPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        if self._needs_fallback():
            return self.fallback.create_att_decode_state(infer_state)
        return NVFP4FlashInferDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


class NVFP4FlashInferPrefillAttState(FlashInferPrefillAttState):
    def init_state(self):
        super().init_state()
        self.triton_state = self.backend.fallback.create_att_prefill_state(self.infer_state)
        self.triton_state.init_state()

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        if att_control.use_sliding_window:
            return self.triton_state.prefill_att(q, k, v, att_control, alloc_func)
        return super().prefill_att(q, k, v, att_control, alloc_func)

    def _nomarl_prefill_att(self, q, k, v, alloc_func=torch.empty):
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        cache, scales = nvfp4_paged_views(k, v, self.backend.infer_page_size)
        return self.prefill_wrapper.run(q, cache, kv_cache_sf=scales, out=out)


class NVFP4FlashInferDecodeAttState(FlashInferDecodeAttState):
    def init_state(self):
        super().init_state()
        if self.decode_wrapper is not None:
            from .nvfp4_decode import use_small_tile_nvfp4_kernel

            use_small_tile_nvfp4_kernel(self.decode_wrapper, self.backend.q_data_type, self.backend.head_dim)
        self.triton_state = self.backend.fallback.create_att_decode_state(self.infer_state)
        self.triton_state.init_state()

    def copy_for_decode_cuda_graph(self, new_state):
        super().copy_for_decode_cuda_graph(new_state)
        self.triton_state.copy_for_decode_cuda_graph(new_state.triton_state)

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        if att_control.use_sliding_window:
            return self.triton_state.decode_att(q, k, v, att_control, alloc_func)
        return super().decode_att(q, k, v, att_control, alloc_func)

    def _normal_decode_att(self, q, k, v, alloc_func=torch.empty):
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        cache, scales = nvfp4_paged_views(k, v, self.backend.infer_page_size)
        return self.decode_wrapper.run(q, cache, kv_cache_sf=scales, out=out)
