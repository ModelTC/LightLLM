import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.envs_utils import get_env_start_args
from .hyper_connection import hc_pre, hc_post
from ..triton_kernel.rotary_emb import apply_rotary_emb
from .compressor import compressor_prefill_state, compressor_decode_step
from .attention import vllm_sparse_attn


DSV4_DEBUG_DIRECT_PREFILL_COMP = os.getenv("DSV4_DEBUG_DIRECT_PREFILL_COMP", "0") == "1"
DSV4_DEBUG_DISABLE_COMP_ATTN = os.getenv("DSV4_DEBUG_DISABLE_COMP_ATTN", "0") == "1"


class DeepseekV4TransformerLayerInfer(TransformerLayerInferTpl):
    """One V4 decoder layer: HC(attn) then HC(ffn).

    The residual is carried as ``hc_mult`` streams flattened to [T, hc_mult*hidden]; each sub-layer
    collapses (hc_pre), computes, and re-expands (hc_post). Attention is MLA over a sliding window +
    compressed KV with a per-head sink (vLLM FlashMLA sparse); the MoE reuses lightllm's deepgemm FP8
    grouped GEMM driven by V4's custom router (sqrtsoftplus + hash/topk + bias-for-selection).

    Per-request decode state (window KV history + compressed KV + compressor running state) is kept in
    DeepseekV4ReqManager so request alloc/free owns its lifetime.
    """

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
        self.topk = cfg["num_experts_per_tok"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.swiglu_limit = cfg["swiglu_limit"]
        self.softmax_scale = self.head_dim ** -0.5
        self.tp_q_heads = self.n_heads // self.tp_world_size_
        self.tp_index_heads = self.index_n_heads // self.tp_world_size_
        self.tp_groups = self.o_groups // self.tp_world_size_
        self.embed_dim_ = self.hc_mult * self.hidden
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self.indexer_score_scale = self.index_head_dim ** -0.5
        self.indexer_weight_scale = self.indexer_score_scale * self.index_n_heads ** -0.5

    # ------------------------------------------------------------------ forward (HC-wrapped)
    def _hc_block(self, streams, infer_state, lw, attn_fn):
        residual = streams
        collapsed, post, comb = hc_pre(
            streams,
            lw.hc_attn_fn_.weight,
            lw.hc_attn_scale_.weight,
            lw.hc_attn_base_.weight,
            self.hc_mult,
            self.hidden,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        o = attn_fn(lw.attn_norm_(collapsed, eps=self.eps_), infer_state, lw)
        streams = hc_post(o, residual, post, comb, self.hc_mult, self.hidden)

        residual = streams
        collapsed, post, comb = hc_pre(
            streams,
            lw.hc_ffn_fn_.weight,
            lw.hc_ffn_scale_.weight,
            lw.hc_ffn_base_.weight,
            self.hc_mult,
            self.hidden,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        f = self._moe_ffn(lw.ffn_norm_(collapsed, eps=self.eps_), infer_state, lw)
        return hc_post(f, residual, post, comb, self.hc_mult, self.hidden)

    def context_forward(self, streams, infer_state, lw):
        return self._hc_block(streams, infer_state, lw, self._attention_prefill)

    def token_forward(self, streams, infer_state, lw):
        return self._hc_block(streams, infer_state, lw, self._attention_decode)

    # ------------------------------------------------------------------ shared projections
    def _qkv(self, x, cos_tok, sin_tok, lw):
        T = x.shape[0]
        qa = lw.q_norm_(lw.wq_a_.mm(x), eps=self.eps_)
        q = lw.wq_b_.mm(qa).view(T, self.tp_q_heads, self.head_dim).float()
        q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps_)).to(x.dtype)
        q = torch.cat(
            [
                q[..., : -self.rope_dim],
                apply_rotary_emb(q[..., -self.rope_dim :], cos_tok.unsqueeze(1), sin_tok.unsqueeze(1)),
            ],
            dim=-1,
        )
        kv = lw.kv_norm_(lw.wkv_.mm(x), eps=self.eps_)
        kv = torch.cat(
            [
                kv[:, : -self.rope_dim],
                apply_rotary_emb(kv[:, -self.rope_dim :], cos_tok, sin_tok),
            ],
            dim=1,
        )
        return q, kv, qa

    def _out_proj(self, o, infer_state, lw):
        # o: [T, tp_q_heads, head_dim] -> inverse rope -> grouped low-rank O -> [T, hidden]
        T = o.shape[0]
        o = o.reshape(T, self.tp_groups, -1).transpose(0, 1).contiguous()  # [groups, T, per_group_in]
        o = lw.wo_a_.bmm(o).transpose(0, 1).reshape(T, -1)  # [T, groups*o_lora]
        o = lw.wo_b_.mm(o)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return o

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

    def _post_dense_kv(self, infer_state, req, start_pos, mem_index, kv):
        positions = torch.arange(
            start_pos,
            start_pos + kv.shape[0],
            device=mem_index.device,
            dtype=torch.long,
        )
        infer_state.mem_manager.pack_mla_kv_to_cache(
            layer_index=self.layer_num_,
            mem_index=mem_index,
            kv=kv.reshape(kv.shape[0], 1, kv.shape[-1]),
            req_idx=req,
            positions=positions,
        )
        return

    def _write_compressed_kv(self, infer_state, req, entry_start, comp):
        slots = infer_state.req_manager.ensure_compress_slots(self.layer_num_, req, entry_start, comp.shape[0])
        if comp.shape[0] == 0:
            return slots
        infer_state.mem_manager.pack_compressed_kv_to_cache(self.layer_num_, slots, comp)
        return slots

    def _write_c4_indexer_k(self, infer_state, slots, idx_comp):
        if idx_comp is None or idx_comp.shape[0] == 0:
            return
        infer_state.mem_manager.pack_c4_indexer_k_to_cache(self.layer_num_, slots, idx_comp)
        return

    def _dense_kv_from_cache(self, infer_state, req, start_pos, end_pos):
        if end_pos <= start_pos:
            return torch.empty((0, self.head_dim), dtype=infer_state.mem_manager.dtype, device="cuda")
        slots = infer_state.req_manager.req_to_token_indexs[req, start_pos:end_pos].long()
        return infer_state.mem_manager.gather_mla_kv(self.layer_num_, slots)

    def _compressed_kv_from_cache(self, infer_state, req, ncomp):
        if ncomp == 0:
            return torch.empty((0, self.head_dim), dtype=infer_state.mem_manager.dtype, device="cuda")
        if self.compress_ratio == 4:
            slots = infer_state.req_manager.req_to_c4_indexs[req, :ncomp].long()
        else:
            slots = infer_state.req_manager.req_to_c128_indexs[req, :ncomp].long()
        return infer_state.mem_manager.gather_compressed_kv(self.layer_num_, slots)

    def _c4_indexer_k_from_cache(self, infer_state, req, ncomp):
        if self.compress_ratio != 4 or ncomp == 0:
            return None
        slots = infer_state.req_manager.req_to_c4_indexs[req, :ncomp].long()
        return infer_state.mem_manager.gather_c4_indexer_k(self.layer_num_, slots)

    # ------------------------------------------------------------------ attention (prefill)
    def _attention_prefill(self, x, infer_state, lw):
        T = x.shape[0]
        if self.compress_ratio:
            cos_tok, sin_tok = (
                infer_state.position_cos_compress,
                infer_state.position_sin_compress,
            )
        else:
            cos_tok, sin_tok = (
                infer_state.position_cos_sliding,
                infer_state.position_sin_sliding,
            )
        q, kv, qa = self._qkv(x, cos_tok, sin_tok, lw)
        sink = lw.attn_sink_.weight
        o = x.new_empty(T, self.tp_q_heads, self.head_dim)
        b_req = infer_state.b_req_idx.tolist()
        starts = infer_state.b_q_start_loc.tolist()
        lens = infer_state.b_q_seq_len.tolist()
        ready_lens = infer_state.b_ready_cache_len.tolist()
        idx_q, idx_weight = self._indexer_q_weight(
            x,
            qa,
            infer_state.position_cos_compress,
            infer_state.position_sin_compress,
            lw,
        )
        for req, st, ln, ready_len in zip(b_req, starts, lens, ready_lens):
            q_r, kv_r, x_r = q[st : st + ln], kv[st : st + ln], x[st : st + ln]
            idx_q_r = None if idx_q is None else idx_q[st : st + ln]
            idx_weight_r = None if idx_weight is None else idx_weight[st : st + ln]
            kv_all, dense_base, n_window, ncomp, idx_comp = self._gather_prefill(
                x_r, kv_r, req, ready_len, lw, infer_state
            )
            ti = self._topk_idxs_prefill(
                ln,
                dense_base,
                n_window,
                ncomp,
                x.device,
                ready_len,
                idx_q_r,
                idx_comp,
                idx_weight_r,
                infer_state,
            )
            o[st : st + ln] = vllm_sparse_attn(q_r.unsqueeze(0), kv_all.unsqueeze(0), sink, ti, self.softmax_scale)[0]
            self._post_dense_kv(
                infer_state,
                req,
                ready_len,
                infer_state.mem_index[st : st + ln],
                kv_r,
            )
        return self._out_proj(self._inv_rope(o, cos_tok, sin_tok), infer_state, lw)

    def _gather_prefill(self, x_r, kv_r, req, ready_len, lw, infer_state):
        ln = kv_r.shape[0]
        idx_comp = None
        if ready_len > 0:
            return self._gather_prefill_extend(x_r, kv_r, req, ready_len, lw, infer_state)
        if self.compress_ratio:
            cstate_pool = infer_state.req_manager.get_compress_state_pool_for_req(self.layer_num_, req)
            comp, ks, ss, cstate_pool = compressor_prefill_state(
                x_r,
                lw.compressor_wkv_.mm_param.weight,
                lw.compressor_wgate_.mm_param.weight,
                lw.compressor_norm_.weight,
                lw.compressor_ape_.weight,
                self.compress_ratio,
                self.head_dim,
                self.rope_dim,
                infer_state.cos_compress_table,
                infer_state.sin_compress_table,
                self.eps_,
                return_state_pool=True,
                state_pool=cstate_pool,
            )
            comp_slots = self._write_compressed_kv(infer_state, req, 0, comp)
            (
                cstate_kv,
                cstate_score,
            ) = infer_state.req_manager.get_compress_state_for_req(self.layer_num_, req)
            cstate_kv.copy_(ks)
            cstate_score.copy_(ss)
            state = {
                "cstate_kv": cstate_kv,
                "cstate_score": cstate_score,
            }
            if self.compress_ratio == 4:
                idx_cstate_pool = infer_state.req_manager.get_c4_indexer_state_pool_for_req(self.layer_num_, req)
                idx_comp, idx_ks, idx_ss, idx_cstate_pool = compressor_prefill_state(
                    x_r,
                    lw.idx_cmp_wkv_.mm_param.weight,
                    lw.idx_cmp_wgate_.mm_param.weight,
                    lw.idx_cmp_norm_.weight,
                    lw.idx_cmp_ape_.weight,
                    4,
                    self.index_head_dim,
                    self.rope_dim,
                    infer_state.cos_compress_table,
                    infer_state.sin_compress_table,
                    self.eps_,
                    return_state_pool=True,
                    state_pool=idx_cstate_pool,
                )
                self._write_c4_indexer_k(infer_state, comp_slots, idx_comp)
                idx_state = infer_state.req_manager.get_c4_indexer_compress_state(self.layer_num_)
                idx_cstate_kv = idx_state[req, 0]
                idx_cstate_score = idx_state[req, 1]
                idx_cstate_kv.copy_(idx_ks)
                idx_cstate_score.copy_(idx_ss)
                state.update(
                    {
                        "idx_cstate_kv": idx_cstate_kv,
                        "idx_cstate_score": idx_cstate_score,
                    }
                )
            infer_state.req_manager.set_runtime_state(
                req,
                self.layer_num_,
                state,
            )
            ncomp = comp.shape[0]
            if DSV4_DEBUG_DISABLE_COMP_ATTN:
                return kv_r, 0, ln, 0, None
            if not DSV4_DEBUG_DIRECT_PREFILL_COMP:
                comp = self._compressed_kv_from_cache(infer_state, req, ncomp)
                idx_comp = self._c4_indexer_k_from_cache(infer_state, req, ncomp)
            return torch.cat([kv_r, comp], dim=0), 0, ln, ncomp, idx_comp
        return kv_r, 0, ln, 0, None

    def _gather_prefill_extend(self, x_r, kv_r, req, ready_len, lw, infer_state):
        if self.compress_ratio:
            try:
                state = infer_state.req_manager.get_runtime_state(req, self.layer_num_)
            except KeyError as exc:
                raise RuntimeError(
                    "DeepSeek-V4 prefill chunk is missing runtime state; radix prompt cache "
                    "must stay disabled until V4 managed token cache is implemented."
                ) from exc
            cstate_pool = infer_state.req_manager.get_compress_state_pool_for_req(self.layer_num_, req)
            idx_cstate_pool = (
                infer_state.req_manager.get_c4_indexer_state_pool_for_req(self.layer_num_, req)
                if self.compress_ratio == 4
                else None
            )

            for j in range(x_r.shape[0]):
                start_pos = ready_len + j
                entry = compressor_decode_step(
                    x_r[j],
                    lw.compressor_wkv_.mm_param.weight,
                    lw.compressor_wgate_.mm_param.weight,
                    lw.compressor_norm_.weight,
                    lw.compressor_ape_.weight,
                    self.compress_ratio,
                    self.head_dim,
                    self.rope_dim,
                    infer_state.cos_compress_table,
                    infer_state.sin_compress_table,
                    self.eps_,
                    state["cstate_kv"],
                    state["cstate_score"],
                    start_pos,
                    state_pool=cstate_pool,
                )
                if entry is not None:
                    entry_start = (start_pos + 1) // self.compress_ratio - 1
                    slots = self._write_compressed_kv(infer_state, req, entry_start, entry.unsqueeze(0))
                if self.compress_ratio == 4:
                    idx_entry = compressor_decode_step(
                        x_r[j],
                        lw.idx_cmp_wkv_.mm_param.weight,
                        lw.idx_cmp_wgate_.mm_param.weight,
                        lw.idx_cmp_norm_.weight,
                        lw.idx_cmp_ape_.weight,
                        4,
                        self.index_head_dim,
                        self.rope_dim,
                        infer_state.cos_compress_table,
                        infer_state.sin_compress_table,
                        self.eps_,
                        state["idx_cstate_kv"],
                        state["idx_cstate_score"],
                        start_pos,
                        state_pool=idx_cstate_pool,
                    )
                    if idx_entry is not None:
                        if entry is None:
                            entry_start = (start_pos + 1) // self.compress_ratio - 1
                            slots = infer_state.req_manager.ensure_compress_slots(self.layer_num_, req, entry_start, 1)
                        self._write_c4_indexer_k(infer_state, slots, idx_entry.unsqueeze(0))
            dense_end = ready_len + x_r.shape[0]
            ncomp = dense_end // self.compress_ratio
            dense_base = max(0, ready_len - self.window + 1)
            cached_dense = self._dense_kv_from_cache(infer_state, req, dense_base, ready_len)
            dense = torch.cat([cached_dense, kv_r], dim=0)
            comp = self._compressed_kv_from_cache(infer_state, req, ncomp)
            idx_comp = self._c4_indexer_k_from_cache(infer_state, req, ncomp)
            if DSV4_DEBUG_DISABLE_COMP_ATTN:
                return dense, dense_base, dense.shape[0], 0, None
            return (
                torch.cat([dense, comp], dim=0),
                dense_base,
                dense.shape[0],
                ncomp,
                idx_comp,
            )
        dense_base = max(0, ready_len - self.window + 1)
        cached_dense = self._dense_kv_from_cache(infer_state, req, dense_base, ready_len)
        dense = torch.cat([cached_dense, kv_r], dim=0)
        return (
            dense,
            dense_base,
            dense.shape[0],
            0,
            None,
        )

    def _topk_idxs_prefill(
        self,
        seqlen,
        dense_base,
        n_window,
        ncomp,
        device,
        base_pos,
        idx_q,
        idx_comp,
        idx_weight,
        infer_state,
    ):
        t = torch.arange(seqlen, device=device)
        abs_pos = t + base_pos
        offsets = torch.arange(self.window, device=device)
        win_abs = abs_pos.unsqueeze(1) - (self.window - 1 - offsets).unsqueeze(0)
        valid = (win_abs >= dense_base) & (win_abs < dense_base + n_window)
        win = torch.where(valid, win_abs - dense_base, torch.full_like(win_abs, -1))
        if ncomp:
            if self.compress_ratio == 4 and ncomp > self.index_topk:
                comp = self._indexer_topk(idx_q, idx_comp, idx_weight, abs_pos + 1, n_window, infer_state)
            else:
                c = torch.arange(ncomp, device=device)
                comp = torch.where(
                    c.unsqueeze(0) < ((abs_pos.unsqueeze(1) + 1) // self.compress_ratio),
                    (c.unsqueeze(0) + n_window).expand(seqlen, ncomp),
                    torch.full((seqlen, ncomp), -1, device=device, dtype=torch.long),
                )
            return torch.cat([win, comp], dim=1).int().unsqueeze(0)
        return win.int().unsqueeze(0)

    # ------------------------------------------------------------------ attention (decode)
    def _attention_decode(self, x, infer_state, lw):
        B = x.shape[0]  # one new token per request
        if self.compress_ratio:
            cos_tok, sin_tok = (
                infer_state.position_cos_compress,
                infer_state.position_sin_compress,
            )
        else:
            cos_tok, sin_tok = (
                infer_state.position_cos_sliding,
                infer_state.position_sin_sliding,
            )
        q, kv, qa = self._qkv(x, cos_tok, sin_tok, lw)  # [B, heads, hd], [B, hd]
        idx_q, idx_weight = self._indexer_q_weight(
            x,
            qa,
            infer_state.position_cos_compress,
            infer_state.position_sin_compress,
            lw,
        )
        sink = lw.attn_sink_.weight
        b_req = infer_state.b_req_idx.tolist()
        seqlens = infer_state.b_seq_len.tolist()
        o = x.new_empty(B, self.tp_q_heads, self.head_dim)
        for i, (req, seq) in enumerate(zip(b_req, seqlens)):
            start_pos = seq - 1
            self._post_dense_kv(
                infer_state,
                req,
                start_pos,
                infer_state.mem_index[i : i + 1],
                kv[i : i + 1],
            )
            if self.compress_ratio:
                try:
                    stt = infer_state.req_manager.get_runtime_state(req, self.layer_num_)
                except KeyError as exc:
                    raise RuntimeError(
                        "DeepSeek-V4 decode is missing runtime state; radix prompt cache "
                        "must stay disabled until V4 managed token cache is implemented."
                    ) from exc
                cstate_pool = infer_state.req_manager.get_compress_state_pool_for_req(self.layer_num_, req)
                e = compressor_decode_step(
                    x[i],
                    lw.compressor_wkv_.mm_param.weight,
                    lw.compressor_wgate_.mm_param.weight,
                    lw.compressor_norm_.weight,
                    lw.compressor_ape_.weight,
                    self.compress_ratio,
                    self.head_dim,
                    self.rope_dim,
                    infer_state.cos_compress_table,
                    infer_state.sin_compress_table,
                    self.eps_,
                    stt["cstate_kv"],
                    stt["cstate_score"],
                    start_pos,
                    state_pool=cstate_pool,
                )
                entry_slots = None
                if e is not None:
                    entry_start = (start_pos + 1) // self.compress_ratio - 1
                    entry_slots = self._write_compressed_kv(infer_state, req, entry_start, e.unsqueeze(0))
                if self.compress_ratio == 4:
                    idx_cstate_pool = infer_state.req_manager.get_c4_indexer_state_pool_for_req(self.layer_num_, req)
                    idx_e = compressor_decode_step(
                        x[i],
                        lw.idx_cmp_wkv_.mm_param.weight,
                        lw.idx_cmp_wgate_.mm_param.weight,
                        lw.idx_cmp_norm_.weight,
                        lw.idx_cmp_ape_.weight,
                        4,
                        self.index_head_dim,
                        self.rope_dim,
                        infer_state.cos_compress_table,
                        infer_state.sin_compress_table,
                        self.eps_,
                        stt["idx_cstate_kv"],
                        stt["idx_cstate_score"],
                        start_pos,
                        state_pool=idx_cstate_pool,
                    )
                    if idx_e is not None:
                        if entry_slots is None:
                            entry_start = (start_pos + 1) // self.compress_ratio - 1
                            entry_slots = infer_state.req_manager.ensure_compress_slots(
                                self.layer_num_, req, entry_start, 1
                            )
                        self._write_c4_indexer_k(infer_state, entry_slots, idx_e.unsqueeze(0))
                win_start = max(0, seq - self.window)
                win_kv = self._dense_kv_from_cache(infer_state, req, win_start, seq)
                comp_kv = self._compressed_kv_from_cache(infer_state, req, seq // self.compress_ratio)
                idx_comp = self._c4_indexer_k_from_cache(infer_state, req, comp_kv.shape[0])
                if DSV4_DEBUG_DISABLE_COMP_ATTN:
                    comp_kv = None
                    idx_comp = None
                    kv_all = win_kv
                else:
                    kv_all = torch.cat([win_kv, comp_kv], dim=0)
            else:
                win_start = max(0, seq - self.window)
                win_kv = self._dense_kv_from_cache(infer_state, req, win_start, seq)
                kv_all = win_kv
                comp_kv = None
                idx_comp = None
            ti = self._topk_idxs_decode(
                win_kv.shape[0],
                comp_kv,
                None if idx_q is None else idx_q[i : i + 1],
                idx_comp,
                None if idx_weight is None else idx_weight[i : i + 1],
                seq,
                x.device,
                infer_state,
            )
            o[i] = vllm_sparse_attn(
                q[i].view(1, 1, self.tp_q_heads, self.head_dim),
                kv_all.unsqueeze(0),
                sink,
                ti,
                self.softmax_scale,
            )[0, 0]
        return self._out_proj(self._inv_rope(o, cos_tok, sin_tok), infer_state, lw)

    def _indexer_q_weight(self, x, qa, cos_tok, sin_tok, lw):
        if self.compress_ratio != 4:
            return None, None
        idx_q = lw.idx_wq_b_.mm(qa).view(x.shape[0], self.tp_index_heads, self.index_head_dim)
        idx_q = torch.cat(
            [
                idx_q[..., : -self.rope_dim],
                apply_rotary_emb(
                    idx_q[..., -self.rope_dim :],
                    cos_tok.unsqueeze(1),
                    sin_tok.unsqueeze(1),
                ),
            ],
            dim=-1,
        )
        idx_weight = lw.idx_weights_proj_.mm(x).float() * self.indexer_weight_scale
        return idx_q, idx_weight

    def _indexer_topk(self, idx_q, idx_comp, idx_weight, positions_1based, offset, infer_state):
        ncomp = idx_comp.shape[0]
        k = min(self.index_topk, ncomp)
        if k == 0:
            return torch.empty((idx_q.shape[0], 0), device=idx_q.device, dtype=torch.long)

        top_chunks = []
        heads = max(1, idx_q.shape[1])
        max_score_elems = 16 * 1024 * 1024
        chunk_size = max(1, min(idx_q.shape[0], max_score_elems // max(1, heads * ncomp)))
        for start in range(0, idx_q.shape[0], chunk_size):
            end = min(idx_q.shape[0], start + chunk_size)
            scores = torch.einsum("thd,nd->thn", idx_q[start:end].float(), idx_comp.float())
            scores = F.relu(scores) * self.indexer_score_scale
            index_scores = (scores * idx_weight[start:end].unsqueeze(-1)).sum(dim=1)
            if self.tp_world_size_ > 1:
                all_reduce(
                    index_scores,
                    op=dist.ReduceOp.SUM,
                    group=infer_state.dist_group,
                    async_op=False,
                )
            causal_threshold = positions_1based[start:end] // 4
            top_chunks.append(self._indexer_topk_kernel(index_scores, causal_threshold, k))
        top = torch.cat(top_chunks, dim=0)
        valid = top >= 0
        return torch.where(valid, top + offset, torch.full_like(top, -1))

    def _indexer_topk_kernel(self, index_scores, causal_threshold, topk):
        if index_scores.is_cuda:
            try:
                import vllm._C  # noqa: F401

                scores = index_scores.contiguous()
                lengths = causal_threshold.to(torch.int32).contiguous()
                starts = torch.zeros_like(lengths, dtype=torch.int32)
                top = torch.empty((scores.shape[0], topk), dtype=torch.int32, device=scores.device)
                torch.ops._C.top_k_per_row_prefill(
                    scores,
                    starts,
                    lengths,
                    top,
                    scores.shape[0],
                    scores.stride(0),
                    scores.stride(1),
                    topk,
                )
                return top.long()
            except Exception:
                pass

        entry_indices = torch.arange(index_scores.shape[1], device=index_scores.device)
        index_scores = index_scores.masked_fill(
            entry_indices.unsqueeze(0) >= causal_threshold.unsqueeze(1), float("-inf")
        )
        top = index_scores.topk(topk, dim=-1).indices
        valid = top < causal_threshold.unsqueeze(1)
        return torch.where(valid, top, torch.full_like(top, -1))

    def _topk_idxs_decode(
        self,
        win_len,
        comp_kv,
        idx_q,
        idx_comp,
        idx_weight,
        seq_len,
        device,
        infer_state,
    ):
        win = torch.arange(win_len, device=device, dtype=torch.long)
        if comp_kv is None or comp_kv.shape[0] == 0:
            return win.view(1, 1, -1).int()
        ncomp = comp_kv.shape[0]
        if self.compress_ratio == 4 and ncomp > self.index_topk:
            comp = self._indexer_topk(
                idx_q,
                idx_comp,
                idx_weight,
                torch.tensor([seq_len], device=device, dtype=torch.long),
                win_len,
                infer_state,
            )[0]
        else:
            comp = torch.arange(ncomp, device=device, dtype=torch.long) + win_len
        return torch.cat([win, comp], dim=0).view(1, 1, -1).int()

    # ------------------------------------------------------------------ moe
    def _fp4_experts(self, x, weights, indices, lw):
        experts = lw.experts_
        if getattr(experts, "moe_backend", None) != "marlin":
            err = getattr(experts, "moe_backend_error", "unknown")
            raise RuntimeError(f"DeepSeek-V4 FP4 MoE requires vLLM Marlin backend, init_error={err}")
        return self._fp4_experts_marlin(x, weights, indices, experts)

    def _fp4_experts_marlin(self, x, weights, indices, experts):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            fused_marlin_moe,
        )
        from vllm.scalar_type import scalar_types

        return fused_marlin_moe(
            hidden_states=x.contiguous(),
            w1=experts.marlin_w13,
            w2=experts.marlin_w2,
            bias1=None,
            bias2=None,
            w1_scale=experts.marlin_w13_scale,
            w2_scale=experts.marlin_w2_scale,
            topk_weights=weights.to(torch.float32).contiguous(),
            topk_ids=indices.to(torch.long).contiguous(),
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=experts.n_routed_experts,
            activation=MoEActivation.SILU,
            clamp_limit=float(self.swiglu_limit),
        )

    def _moe_ffn(self, x, infer_state, lw):
        gw = lw.gate_weight_.mm_param.weight
        logits = F.linear(x.float(), gw.float()).contiguous()
        weights, indices = self._select_experts(logits, infer_state, lw)
        routed = self._fp4_experts(x, weights, indices, lw)
        g = lw.shared_gate_.mm(x).float().clamp(max=self.swiglu_limit)
        u = lw.shared_up_.mm(x).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        shared = lw.shared_down_.mm((F.silu(g) * u).to(x.dtype))
        if self.enable_ep_moe and getattr(lw.experts_, "is_ep", False):
            if self.tp_world_size_ > 1:
                all_reduce(
                    shared,
                    op=dist.ReduceOp.SUM,
                    group=infer_state.dist_group,
                    async_op=False,
                )
            return routed + shared
        out = routed + shared
        if self.tp_world_size_ > 1:
            all_reduce(out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return out

    def _select_experts(self, logits, infer_state, lw):
        return self._select_experts_vllm(logits, infer_state, lw)

    def _select_experts_vllm(self, logits, infer_state, lw):
        from vllm import _custom_ops as ops

        M = logits.shape[0]
        bias = None
        input_tokens = None
        hash_indices_table = None
        indices_dtype = torch.int64
        if self.is_hash:
            hash_indices_table = lw.gate_tid2eid_.weight
            if not hash_indices_table.is_contiguous():
                hash_indices_table = hash_indices_table.contiguous()
            indices_dtype = hash_indices_table.dtype
            input_tokens = infer_state.input_ids.to(dtype=indices_dtype).contiguous()
        else:
            bias = lw.gate_bias_.weight

        weights = torch.empty((M, self.topk), dtype=torch.float32, device=logits.device)
        indices = torch.empty((M, self.topk), dtype=indices_dtype, device=logits.device)
        token_expert_indices = torch.empty((M, self.topk), dtype=torch.int32, device=logits.device)
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
