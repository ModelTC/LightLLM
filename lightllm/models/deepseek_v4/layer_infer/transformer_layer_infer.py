import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from .hyper_connection import hc_pre, hc_fused_post_pre, hc_post
from .compressor import (
    compressor_prefill_state,
    compressor_decode_step_single,
    compressor_decode_step_batch,
    compressor_paged_prefill,
    compressor_paged_decode_batch,
    paged_prefill_compress_data,
    paged_decode_state_slots,
)
from ..triton_kernel.rotary_emb import apply_rotary_emb
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4TransformerLayerInfer(TransformerLayerInferTpl):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        cfg = network_config
        self.eps_ = cfg["rms_norm_eps"]
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_dim = cfg["qk_rope_head_dim"]
        self.index_n_heads = cfg["index_n_heads"]
        self.index_head_dim = cfg["index_head_dim"]
        self.index_topk = cfg["index_topk"]
        self.o_groups = cfg["o_groups"]
        self.o_lora = cfg["o_lora_rank"]
        self.hc_mult = cfg["hc_mult"]
        self.sinkhorn_iters = cfg["hc_sinkhorn_iters"]
        self.hc_eps = cfg["hc_eps"]
        self.window = cfg["sliding_window"]
        self.compress_ratio = cfg["compress_ratios"][layer_num]
        self.is_hash = layer_num < cfg["num_hash_layers"]
        self.is_last_layer = layer_num == cfg["n_layer"] - 1
        # complex64 rope table for this layer's variant (sliding / compressed); set by
        # DeepseekV4TpPartModel._init_to_get_rotary once the tables are built. The full compress
        # cos/sin tables (compressor entry rope uses entry positions, not token positions) are
        # wired there too.
        self.freqs_cis = None
        self.cos_compress_table = None
        self.sin_compress_table = None
        self.topk = cfg["num_experts_per_tok"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.swiglu_limit = cfg["swiglu_limit"]
        self.softmax_scale = self.head_dim ** -0.5
        self.tp_q_heads = self.n_heads // self.tp_world_size_
        self.tp_index_heads = self.index_n_heads // self.tp_world_size_
        self.tp_groups = self.o_groups // self.tp_world_size_
        self.tp_q_head_num_ = self.tp_q_heads
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.tp_o_head_num_ = self.tp_q_heads
        self.head_dim_ = self.head_dim
        self.embed_dim_ = self.hc_mult * self.hidden
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self.indexer_score_scale = self.index_head_dim ** -0.5
        self.indexer_weight_scale = self.indexer_score_scale * self.index_n_heads ** -0.5

    # ------------------------------------------------------------------ forward (HC-threaded)
    def _hc_attn_in(self, input_embdings, layer_weight: DeepseekV4TransformerLayerWeight):
        """Layer input -> attention input (attn_norm fused). First layer gets the raw streams
        and runs a standalone hc_pre; later layers get (x, residual, post_mix, res_mix) and fuse
        the previous layer's ffn hc_post with this layer's attn hc_pre."""
        if torch.is_tensor(input_embdings):
            residual = input_embdings.view(-1, self.hc_mult, self.hidden)
            return hc_pre(
                residual,
                layer_weight.hc_attn_fn_.weight,
                layer_weight.hc_attn_scale_.weight,
                layer_weight.hc_attn_base_.weight,
                self.eps_,
                self.hc_eps,
                self.sinkhorn_iters,
                layer_weight.attn_norm_.weight,
                self.eps_,
            )
        x, residual, post_mix, res_mix = input_embdings
        return hc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            layer_weight.hc_attn_fn_.weight,
            layer_weight.hc_attn_scale_.weight,
            layer_weight.hc_attn_base_.weight,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            layer_weight.attn_norm_.weight,
            self.eps_,
        )

    def _hc_ffn_in(self, x, residual, post_mix, res_mix, layer_weight: DeepseekV4TransformerLayerWeight):
        """Attention output -> ffn input (ffn_norm fused): fused attn hc_post + ffn hc_pre."""
        return hc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            layer_weight.hc_ffn_fn_.weight,
            layer_weight.hc_ffn_scale_.weight,
            layer_weight.hc_ffn_base_.weight,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            layer_weight.ffn_norm_.weight,
            self.eps_,
        )

    def _hc_ffn_out(self, x, residual, post_mix, res_mix):
        """Mid layers leave the ffn hc_post pending for the next layer's fused post+pre; the last
        layer completes it and hands the flat streams [T, hc_mult*hidden] back to the model loop."""
        if not self.is_last_layer:
            return x, residual, post_mix, res_mix
        streams = hc_post(x, residual, post_mix, res_mix)
        return streams.reshape(streams.shape[0], -1)

    def context_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, post_mix, res_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.context_attention_forward(x, infer_state, layer_weight)
        x, residual, post_mix, res_mix = self._hc_ffn_in(x, residual, post_mix, res_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, post_mix, res_mix)

    def token_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, post_mix, res_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.token_attention_forward(x, infer_state, layer_weight)
        x, residual, post_mix, res_mix = self._hc_ffn_in(x, residual, post_mix, res_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, post_mix, res_mix)

    # ------------------------------------------------------------------ shared projections / cache
    def _select_rope(self, infer_state: DeepseekV4InferStateInfo):
        if self.compress_ratio:
            return infer_state.position_cos_compress, infer_state.position_sin_compress
        return infer_state.position_cos_sliding, infer_state.position_sin_sliding

    def _get_qkv(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        from sglang.jit_kernel.dsv4 import fused_q_norm_rope

        x = self._tpsp_allgather(input=x, infer_state=infer_state)
        cos_tok, sin_tok = self._select_rope(infer_state)
        T = x.shape[0]
        qa = layer_weight.q_norm_(layer_weight.wq_a_.mm(x), eps=self.eps_)
        q_in = layer_weight.wq_b_.mm(qa).view(T, self.tp_q_heads, self.head_dim)
        # per-(token, head) weightless self-RMSNorm + interleaved rope on the last rope_dim dims,
        # fused in one sglang dsv4 jit kernel (fp32 norm/rotation, bf16 in between -- same as eager).
        q = self.alloc_tensor(q_in.shape, dtype=q_in.dtype, device=q_in.device)
        fused_q_norm_rope(q_in, q, self.eps_, self.freqs_cis, infer_state.position_ids)
        # kv: rmsnorm + rope + fp8 pack + scatter 进 swa 池,一个 sglang jit kernel 完成
        # (同 sglang _compute_kv_to_cache),替代 eager norm/rope/cat + _post_cache_kv。
        # bf16 kv 中间量没有其他消费者: flashmla 路径注意力读 cache,压缩器/indexer 取 x。
        infer_state.mem_manager.pack_mla_kv_to_cache_fused_norm_rope(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=layer_weight.wkv_.mm(x),
            kv_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            positions=infer_state.position_ids,
        )
        return q, qa, cos_tok, sin_tok

    def _get_o(self, o, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        # o: [T, tp_q_heads, head_dim] after inverse rope -> grouped low-rank O -> [T, hidden]
        T = o.shape[0]
        o = o.reshape(T, self.tp_groups, -1).transpose(0, 1).contiguous()  # [groups, T, per_group_in]
        o = layer_weight.wo_a_.bmm(o).transpose(0, 1).reshape(T, -1)  # [T, groups*o_lora]
        o = layer_weight.wo_b_.mm(o)
        return self._tpsp_reduce(input=o, infer_state=infer_state)

    def _inv_rope(self, o, cos_tok, sin_tok):
        return torch.cat(
            [
                o[..., : -self.rope_dim],
                apply_rotary_emb(
                    o[..., -self.rope_dim :],
                    cos_tok.unsqueeze(1),
                    sin_tok.unsqueeze(1),
                    inverse=True,
                ),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ compressor / indexer
    def _indexer_q_weight(
        self, x, q_lora, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        if self.compress_ratio != 4:
            return None, None
        cos_tok = infer_state.position_cos_compress
        sin_tok = infer_state.position_sin_compress
        idx_q = layer_weight.idx_wq_b_.mm(q_lora).view(x.shape[0], self.tp_index_heads, self.index_head_dim)
        idx_q = torch.cat(
            [
                idx_q[..., : -self.rope_dim],
                apply_rotary_emb(idx_q[..., -self.rope_dim :], cos_tok.unsqueeze(1), sin_tok.unsqueeze(1)),
            ],
            dim=-1,
        )
        idx_weight = layer_weight.idx_weights_proj_.mm(x).float() * self.indexer_weight_scale
        return idx_q, idx_weight

    def _gather_compress_slots(self, infer_state: DeepseekV4InferStateInfo, req, entry_start, entry_count):
        """组末 token 的 full 槽位 -> 压缩槽(条目 [entry_start, entry_start+entry_count))。
        槽位已由 prep 阶段(prepare_*_compress_slots)分配并 scatter 进 full_to_c4/c128_indexs。"""
        ratio = self.compress_ratio
        mem = infer_state.mem_manager
        mapping = mem.full_to_c4_indexs if ratio == 4 else mem.full_to_c128_indexs
        last = entry_start + entry_count
        ends = infer_state.req_manager.req_to_token_indexs[req, ratio - 1 : last * ratio : ratio][entry_start:]
        return mapping[ends.long()]

    def _write_compressed_kv(self, infer_state: DeepseekV4InferStateInfo, req, entry_start, comp):
        slots = self._gather_compress_slots(infer_state, req, entry_start, comp.shape[0])
        if comp.shape[0]:
            infer_state.mem_manager.pack_compressed_kv_to_cache(self.layer_num_, slots, comp)
        return slots

    def _compressor_weights(self, layer_weight: DeepseekV4TransformerLayerWeight, for_indexer: bool):
        if for_indexer:
            return (
                layer_weight.idx_cmp_wkv_.mm_param.weight,
                layer_weight.idx_cmp_wgate_.mm_param.weight,
                layer_weight.idx_cmp_norm_.weight,
                layer_weight.idx_cmp_ape_.weight,
                self.index_head_dim,
            )
        return (
            layer_weight.compressor_wkv_.mm_param.weight,
            layer_weight.compressor_wgate_.mm_param.weight,
            layer_weight.compressor_norm_.weight,
            layer_weight.compressor_ape_.weight,
            self.head_dim,
        )

    def _run_compressor_prefill(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        """Per-request compressor for the prefill chunk. Runs as part of the deferred attention
        func, before the attention metadata gathers the slot mappings.

        c4: paged state (swa-page-derived group slots, translation #3) — one fused extend-aware
        call per request; the (write_loc, extra_data, plan) tuple is layer-independent and cached
        on infer_state across all c4 layers. c128: req-keyed state (zero at every 128 boundary by
        construction, nothing cache-resident), original jit paths."""
        if not self.compress_ratio:
            return
        if self.compress_ratio == 4:
            self._run_c4_compressor_prefill(x, infer_state, layer_weight)
        else:
            self._run_c128_compressor_prefill(x, infer_state, layer_weight)
        return

    def _run_c4_compressor_prefill(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        rm = infer_state.req_manager
        mem = infer_state.mem_manager
        wkv, wgate, norm, ape, _ = self._compressor_weights(layer_weight, for_indexer=False)
        iwkv, iwgate, inorm, iape, _ = self._compressor_weights(layer_weight, for_indexer=True)
        state_buf = mem.get_c4_state_buffer(self.layer_num_)
        idx_state_buf = mem.get_c4_indexer_state_buffer(self.layer_num_)
        data_cache = getattr(infer_state, "_dsv4_c4_prefill_data", None)
        if data_cache is None:
            data_cache = {}
            infer_state._dsv4_c4_prefill_data = data_cache
        b_req = infer_state.b_req_idx.tolist()
        starts = infer_state.b_q_start_loc.tolist()
        lens = infer_state.b_q_seq_len.tolist()
        ready_lens = infer_state.b_ready_cache_len.tolist()
        for req, st, ln, ready_len in zip(b_req, starts, lens, ready_lens):
            if req == rm.HOLD_REQUEST_ID or ln == 0:
                continue
            seq_len = ready_len + ln
            data = data_cache.get(req)
            if data is None:
                data = paged_prefill_compress_data(
                    rm.req_to_token_indexs, mem.full_to_swa_indexs, req, ready_len, seq_len, ring=8
                )
                data_cache[req] = data
            x_r = x[st : st + ln]
            comp = compressor_paged_prefill(
                x_r,
                wkv,
                wgate,
                norm,
                ape,
                self.head_dim,
                self.cos_compress_table,
                self.sin_compress_table,
                self.eps_,
                state_buf,
                data,
                ready_len,
                seq_len,
            )
            slots = self._write_compressed_kv(infer_state, req, ready_len // 4, comp)
            idx_comp = compressor_paged_prefill(
                x_r,
                iwkv,
                iwgate,
                inorm,
                iape,
                self.index_head_dim,
                self.cos_compress_table,
                self.sin_compress_table,
                self.eps_,
                idx_state_buf,
                data,
                ready_len,
                seq_len,
            )
            if idx_comp.shape[0]:
                infer_state.mem_manager.pack_indexer_k_to_cache(self.layer_num_, slots, idx_comp)
        return

    def _run_c128_compressor_prefill(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        rm = infer_state.req_manager
        wkv, wgate, norm, ape, _ = self._compressor_weights(layer_weight, for_indexer=False)
        b_req = infer_state.b_req_idx.tolist()
        starts = infer_state.b_q_start_loc.tolist()
        lens = infer_state.b_q_seq_len.tolist()
        ready_lens = infer_state.b_ready_cache_len.tolist()
        for req, st, ln, ready_len in zip(b_req, starts, lens, ready_lens):
            if req == rm.HOLD_REQUEST_ID:
                continue
            x_r = x[st : st + ln]
            state_pool = rm.get_compress_state_pool_for_req(self.layer_num_, req)
            if ready_len == 0:
                comp = compressor_prefill_state(
                    x_r,
                    wkv,
                    wgate,
                    norm,
                    ape,
                    self.compress_ratio,
                    self.head_dim,
                    self.cos_compress_table,
                    self.sin_compress_table,
                    self.eps_,
                    state_pool,
                )
                self._write_compressed_kv(infer_state, req, 0, comp)
            else:
                for j in range(ln):
                    start_pos = ready_len + j
                    entry = compressor_decode_step_single(
                        x_r[j],
                        wkv,
                        wgate,
                        norm,
                        ape,
                        self.compress_ratio,
                        self.head_dim,
                        self.cos_compress_table,
                        self.sin_compress_table,
                        self.eps_,
                        state_pool,
                        start_pos,
                    )
                    if entry is not None:
                        entry_start = (start_pos + 1) // self.compress_ratio - 1
                        self._write_compressed_kv(infer_state, req, entry_start, entry.unsqueeze(0))
        return

    def _run_compressor_decode(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        """Batched decode compressor (cuda-graph safe): state update for every request, cache write
        masked to the pool HOLD slot unless this token completes a window. Compressed-cache slots
        were pre-allocated by prepare_decode_compress_slots in the prep phase.

        c4: paged state — group slots derived from full_to_swa (translation #3) via pure tensor
        ops (graph-safe), shared across all c4 layers per step. c128: req-keyed state."""
        if not self.compress_ratio:
            return
        rm = infer_state.req_manager
        mem = infer_state.mem_manager
        req = infer_state.b_req_idx
        ratio = self.compress_ratio
        wkv, wgate, norm, ape, _ = self._compressor_weights(layer_weight, for_indexer=False)

        if ratio == 4:
            mapping, hold = mem.full_to_c4_indexs, mem.c4_pool.HOLD_TOKEN_MEMINDEX
            slot_meta = getattr(infer_state, "_dsv4_c4_decode_slots", None)
            if slot_meta is None:
                slot_meta = paged_decode_state_slots(
                    rm.req_to_token_indexs,
                    mem.full_to_swa_indexs,
                    req,
                    infer_state.b_seq_len,
                    page_size=128,
                    ring=8,
                    ratio=4,
                    hold_req_id=rm.HOLD_REQUEST_ID,
                    num_swa_pages=mem.swa_num_pages,
                )
                infer_state._dsv4_c4_decode_slots = slot_meta
            write_slot, overlap_slot = slot_meta
            entry, should = compressor_paged_decode_batch(
                x,
                wkv,
                wgate,
                norm,
                ape,
                self.head_dim,
                self.cos_compress_table,
                self.sin_compress_table,
                self.eps_,
                mem.get_c4_state_buffer(self.layer_num_),
                write_slot,
                overlap_slot,
                infer_state.b_seq_len,
            )
        else:
            mapping, hold = mem.full_to_c128_indexs, mem.c128_pool.HOLD_TOKEN_MEMINDEX
            entry, should = compressor_decode_step_batch(
                x,
                wkv,
                wgate,
                norm,
                ape,
                ratio,
                self.head_dim,
                self.rope_dim,
                self.cos_compress_table,
                self.sin_compress_table,
                self.eps_,
                rm.get_compress_state_pool(self.layer_num_),
                req,
                infer_state.b_seq_len.long() - 1,
            )

        should = should & (req != rm.HOLD_REQUEST_ID)
        # 本步 token 即组末 token(should 为真时)，其 full 槽 = mem_index，映射在 prep 已 scatter。
        slots = mapping[infer_state.mem_index.long()].long()
        slots = torch.where(should, slots, torch.full_like(slots, hold))
        mem.pack_compressed_kv_to_cache(self.layer_num_, slots, entry)

        if ratio == 4:
            iwkv, iwgate, inorm, iape, _ = self._compressor_weights(layer_weight, for_indexer=True)
            idx_entry, idx_should = compressor_paged_decode_batch(
                x,
                iwkv,
                iwgate,
                inorm,
                iape,
                self.index_head_dim,
                self.cos_compress_table,
                self.sin_compress_table,
                self.eps_,
                mem.get_c4_indexer_state_buffer(self.layer_num_),
                write_slot,
                overlap_slot,
                infer_state.b_seq_len,
            )
            idx_should = idx_should & (req != rm.HOLD_REQUEST_ID)
            idx_slots = torch.where(idx_should, slots, torch.full_like(slots, hold))
            mem.pack_indexer_k_to_cache(self.layer_num_, idx_slots, idx_entry)
        return

    # ------------------------------------------------------------------ attention (prefill)
    def context_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        # _get_qkv writes the chunk's packed latent into the swa pool (fused kernel) before
        # attention reads it back via full_to_swa indices (this custom forward bypasses the
        # tpl _post_cache_kv path).
        q, q_lora, cos_tok, sin_tok = self._get_qkv(x, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(q, q_lora, x, infer_state, layer_weight)
        return self._get_o(self._inv_rope(o, cos_tok, sin_tok), infer_state, layer_weight)

    def _context_attention_wrapper_run(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        if torch.cuda.is_current_stream_capturing():
            q = q.contiguous()
            q_lora = q_lora.contiguous()
            x = x.contiguous()
            _q = tensor_to_no_ref_tensor(q)
            _q_lora = tensor_to_no_ref_tensor(q_lora)
            _x = tensor_to_no_ref_tensor(x)

            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            # Same graph-split output handoff as the template, but avoid its dry-run because
            # DSV4 attention mutates compressor/cache state before returning.
            o = self.alloc_tensor((q.shape[0], self.tp_q_heads, self.head_dim), dtype=q.dtype, device=q.device)
            _o = tensor_to_no_ref_tensor(o)

            def att_func(new_infer_state: DeepseekV4InferStateInfo):
                tmp_o = self._context_attention_kernel(_q, _q_lora, _x, new_infer_state, layer_weight)
                assert tmp_o.shape == _o.shape
                _o.copy_(tmp_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
            return o

        return self._context_attention_kernel(q, q_lora, x, infer_state, layer_weight)

    def _context_attention_kernel(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        self._run_compressor_prefill(x, infer_state, layer_weight)
        idx_q, idx_weight = self._indexer_q_weight(x, q_lora, infer_state, layer_weight)
        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "flashmla_kvcache": True,
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.head_dim,
                "softmax_scale": self.softmax_scale,
                "q_lora": q_lora,
                "hidden_states": x,
                "attn_sink": layer_weight.attn_sink_.weight,
                "idx_q": idx_q,
                "idx_weight": idx_weight,
                "index_topk": self.index_topk,
                "indexer_score_scale": self.indexer_score_scale,
                "tp_world_size": self.tp_world_size_,
            },
        )
        return infer_state.prefill_att_state.prefill_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )

    # ------------------------------------------------------------------ attention (decode)
    def token_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        q, q_lora, cos_tok, sin_tok = self._get_qkv(x, infer_state, layer_weight)
        o = self._token_attention_kernel(q, q_lora, x, infer_state, layer_weight)
        return self._get_o(self._inv_rope(o, cos_tok, sin_tok), infer_state, layer_weight)

    def _token_attention_kernel(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        self._run_compressor_decode(x, infer_state, layer_weight)
        idx_q, idx_weight = self._indexer_q_weight(x, q_lora, infer_state, layer_weight)
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "flashmla_kvcache": True,
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.head_dim,
                "softmax_scale": self.softmax_scale,
                "q_lora": q_lora,
                "hidden_states": x,
                "attn_sink": layer_weight.attn_sink_.weight,
                "idx_q": idx_q,
                "idx_weight": idx_weight,
                "index_topk": self.index_topk,
                "indexer_score_scale": self.indexer_score_scale,
                "tp_world_size": self.tp_world_size_,
            },
        )
        return infer_state.decode_att_state.decode_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )

    # ------------------------------------------------------------------ moe
    def _routed_experts(self, x, weights, indices, layer_weight: DeepseekV4TransformerLayerWeight):
        return layer_weight.experts_.experts_with_preselected(
            input_tensor=x,
            topk_weights=weights,
            topk_ids=indices,
            clamp_limit=float(self.swiglu_limit),
        )

    def _ffn(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        x = x.view(-1, self.hidden)
        if not self.enable_ep_moe:
            x = self._tpsp_allgather(input=x, infer_state=infer_state)

        gw = layer_weight.gate_weight_.mm_param.weight
        logits = F.linear(x.float(), gw.float()).contiguous()
        weights, indices = self._select_experts(logits, infer_state, layer_weight)
        routed = self._routed_experts(x, weights, indices, layer_weight)
        g = layer_weight.shared_gate_.mm(x).float().clamp(max=self.swiglu_limit)
        u = layer_weight.shared_up_.mm(x).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        shared = layer_weight.shared_down_.mm((F.silu(g) * u).to(x.dtype))
        if self.enable_ep_moe:
            if self.tp_world_size_ > 1:
                all_reduce(
                    shared,
                    op=dist.ReduceOp.SUM,
                    group=infer_state.dist_group,
                    async_op=False,
                )
            return routed + shared
        out = routed + shared
        return self._tpsp_reduce(input=out, infer_state=infer_state)

    def _select_experts(
        self, logits, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        return self._select_experts_vllm(logits, infer_state, layer_weight)

    def _select_experts_vllm(
        self, logits, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        from vllm import _custom_ops as ops

        M = logits.shape[0]
        bias = None
        input_tokens = None
        hash_indices_table = None
        indices_dtype = torch.int64
        if self.is_hash:
            hash_indices_table = layer_weight.gate_tid2eid_.weight
            if not hash_indices_table.is_contiguous():
                hash_indices_table = hash_indices_table.contiguous()
            indices_dtype = hash_indices_table.dtype
            input_tokens = infer_state.input_ids.to(dtype=indices_dtype).contiguous()
        else:
            bias = layer_weight.gate_bias_.weight

        weights = self.alloc_tensor((M, self.topk), dtype=torch.float32, device=logits.device)
        indices = self.alloc_tensor((M, self.topk), dtype=indices_dtype, device=logits.device)
        token_expert_indices = self.alloc_tensor((M, self.topk), dtype=torch.int32, device=logits.device)
        ops.topk_hash_softplus_sqrt(
            weights,
            indices,
            token_expert_indices,
            logits,
            True,
            self.route_scale,
            bias,
            input_tokens,
            hash_indices_table,
        )
        return weights, indices.long()
