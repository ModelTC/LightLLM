import torch
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.req_manager import DeepseekV4ReqManager
from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager


class DeepseekV4InferStateInfo(InferStateInfo):
    req_manager: DeepseekV4ReqManager
    mem_manager: DeepseekV4MemoryManager

    """Per-token interleaved-rope cos/sin for the two rope variants (sliding / compressed), following
    the gemma4 two-variant convention (_cos_cached_* -> position_cos_*). The full rope tables are
    model constants and live on the model / layer infers, not here."""

    def __init__(self):
        super().__init__()
        self.position_cos_sliding = None
        self.position_sin_sliding = None
        self.position_cos_compress = None
        self.position_sin_compress = None
        # layer-independent sparse-index metadata, built once per forward in init_some_extra_state
        # (None until then so copy_for_cuda_graph's tensor-attr loop skips them).
        self.dsv4_sparse_req_idx = None
        self.dsv4_swa_indices = None
        self.dsv4_swa_lengths = None
        # token -> batch-position map for the compressor; built per prefill forward in init_some_extra_state.
        self._dsv4_token_to_batch_idx = None
        # lazily-built (first c4 layer) cache of layer-independent paged-c4 metadata; reused by the
        # other c4 layers in the same forward. Plain tuple (not a tensor attr) so copy_for_cuda_graph
        # ignores it -- it's a capture-time wiring of layer0->others, not a staged graph input.
        self._c4_paged_meta = None

    def init_some_extra_state(self, model):
        self._c4_paged_meta = None  # reset per forward before any c4 layer runs
        super().init_some_extra_state(model)  # sets position_ids, b_q_seq_len, b_q_start_loc (prefill)
        pos = self.position_ids
        self.position_cos_sliding = torch.index_select(model._cos_cached_sliding, 0, pos)  # [T, rope_dim//2]
        self.position_sin_sliding = torch.index_select(model._sin_cached_sliding, 0, pos)
        self.position_cos_compress = torch.index_select(model._cos_cached_compress, 0, pos)
        self.position_sin_compress = torch.index_select(model._sin_cached_compress, 0, pos)
        # Per-token request id (decode: one token per req; prefill: ragged -> repeat by q-len).
        # Layer-independent; the swa kernel + build_metadata's c4/c128 readers all reuse it.
        if self.is_prefill:
            self.dsv4_sparse_req_idx = torch.repeat_interleave(self.b_req_idx, self.b_q_seq_len.long())
            self._dsv4_token_to_batch_idx = torch.repeat_interleave(
                torch.arange(self.b_req_idx.shape[0], device=self.b_req_idx.device),
                self.b_q_seq_len.long(),
                output_size=pos.numel(),
            ).to(torch.int32)
        else:
            self.dsv4_sparse_req_idx = self.b_req_idx
            self._dsv4_token_to_batch_idx = None
        # Sliding-window indices: layer-independent (full_to_swa is global, window is const), so build
        # once here via one fused kernel instead of recomputing per layer. const [T, window] shape is
        # cuda-graph-safe (no max_kv_seq_len dependence) and auto-staged by copy_for_cuda_graph.
        from lightllm.models.deepseek_v4.triton_kernel.build_swa_index_dsv4 import build_swa_index

        self.dsv4_swa_indices, self.dsv4_swa_lengths = build_swa_index(
            req_idx=self.dsv4_sparse_req_idx,
            positions=self.position_ids,
            req_to_token_indexs=self.req_manager.req_to_token_indexs,
            full_to_swa_indexs=self.mem_manager.full_to_swa_indexs,
            window=int(self.mem_manager.sliding_window),
        )
        # prefill-cudagraph 桶填充的 HOLD 尾请求的 q 行数。其注意力读 HOLD 槽位(内容被并发写
        # 竞争,每轮不同),输出必须清零,否则 pad 行 hidden 不确定 -> MoE 路由抖动 -> 共享 expert
        # 批次组成变化 -> 真实行 GEMM 归约顺序变化(ulp 级),44 层放大后翻转低置信 token。
        self._dsv4_prefill_pad_q_len = 0
        if self.is_prefill and self.b_req_idx.numel() > 0:
            if int(self.b_req_idx[-1].item()) == self.req_manager.HOLD_REQUEST_ID:
                self._dsv4_prefill_pad_q_len = int((self.b_seq_len[-1] - self.b_ready_cache_len[-1]).item())
