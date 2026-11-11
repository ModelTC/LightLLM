"""NVFP4 prefill: decode each visible KV once, then use FA3 tensor cores."""

import dataclasses
import weakref

import torch

from ..base_att import AttControl, BasePrefillAttState
from ..triton.nvfp4 import NVFP4TritonAttBackend
from ...triton_kernel.kv_copy.dequantize_nvfp4 import dequantize_nvfp4_kv
from lightllm.utils.sgl_utils import flash_attn_varlen_func


class NVFP4Fa3AttBackend(NVFP4TritonAttBackend):
    # K and V together; never allocate a persistent BF16 mirror of the cache.
    max_dequant_workspace_bytes = 256 * 1024 * 1024

    def __init__(self, model):
        super().__init__(model)
        if flash_attn_varlen_func is None or torch.cuda.get_device_capability()[0] < 9:
            raise ValueError("NVFP4 FA3 prefill requires sgl_kernel FA3 and SM90 or newer")

    def create_att_prefill_state(self, infer_state):
        return NVFP4Fa3PrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NVFP4Fa3PrefillAttState(BasePrefillAttState):
    cu_q: torch.Tensor = None
    cu_k: torch.Tensor = None
    paged_state: object = None

    def init_state(self):
        self.cu_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_k = self.infer_state.b1_cu_kv_seq_len.int()

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert not att_control.use_alibi and not att_control.use_att_sink
        # Use the actual local storage shape, including replicated KV heads.
        workspace_bytes = self.infer_state.total_token_num * k.shape[1] * q.shape[-1] * q.element_size() * 2
        if workspace_bytes > self.backend.max_dequant_workspace_bytes and self.paged_state is None:
            # Large shared-prefix batches must not grow a BF16 staging buffer
            # with batch * context. Native paged NVFP4 has fixed scratch space.
            from ..flashinfer.nvfp4 import NVFP4FlashInferAttBackend

            if torch.cuda.is_current_stream_capturing():
                # Prefill graph shape probes can enter here before any eager
                # attention call. FI planning needs CPU metadata and must stay
                # outside capture; the fused fallback is entirely GPU-side.
                self.paged_state = NVFP4TritonAttBackend.create_att_prefill_state(self.backend, self.infer_state)
            else:
                backend = NVFP4FlashInferAttBackend(self.backend.model)
                self.paged_state = backend.create_att_prefill_state(self.infer_state)
            self.paged_state.init_state()
        if self.paged_state is not None:
            return self.paged_state.prefill_att(q, k, v, att_control, alloc_func)
        tokens = self.infer_state.total_token_num
        shape = (tokens, k.shape[1], q.shape[-1])
        kd = alloc_func(shape, dtype=q.dtype, device=q.device)
        vd = alloc_func(shape, dtype=q.dtype, device=q.device)
        dequantize_nvfp4_kv(
            k,
            v,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_seq_len,
            self.cu_k,
            self.infer_state.max_kv_seq_len,
            kd,
            vd,
        )
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return flash_attn_varlen_func(
            q,
            kd,
            vd,
            self.cu_q,
            self.cu_k,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            max_seqlen_k=self.infer_state.max_kv_seq_len,
            causal=self.backend.uses_causal_attention(),
            window_size=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
            out=out,
        )
