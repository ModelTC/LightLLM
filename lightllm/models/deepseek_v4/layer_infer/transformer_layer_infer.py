import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.distributed.communication_op import all_reduce
from lightllm.utils.envs_utils import get_env_start_args
from .hyper_connection import hc_pre, hc_post
from ..triton_kernel.rotary_emb import apply_rotary_emb
from .compressor import compressor_prefill_state, compressor_decode_step
from .attention import torch_sparse_attn
from ..triton_kernel.quant_convert import dequant_fp4_group_to_bf16


class DeepseekV4TransformerLayerInfer(TransformerLayerInferTpl):
    """One V4 decoder layer: HC(attn) then HC(ffn). Correctness-first pure-torch.

    The residual is carried as ``hc_mult`` streams flattened to [T, hc_mult*hidden]; each sub-layer
    collapses (hc_pre), computes, and re-expands (hc_post). Attention is MLA over a sliding window +
    compressed KV with a per-head sink (torch_sparse_attn); the MoE reuses lightllm's deepgemm FP8
    grouped GEMM driven by V4's custom router (sqrtsoftplus + hash/topk + bias-for-selection).

    Per-request decode state (window KV history + compressed KV + compressor running state) is kept in
    a dict keyed by request id. NOTE: correctness-first — this should move into the KV mem manager for
    production memory management / request eviction.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        cfg = network_config
        self.eps_ = cfg["rms_norm_eps"]
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_dim = cfg["qk_rope_head_dim"]
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
        self.softmax_scale = self.head_dim**-0.5
        self.tp_q_heads = self.n_heads // self.tp_world_size_
        self.tp_groups = self.o_groups // self.tp_world_size_
        self.embed_dim_ = self.hc_mult * self.hidden
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self._state = {}  # req_id -> dict(kv_hist, comp_kv, cstate_kv, cstate_score)

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
        kv = torch.cat([kv[:, : -self.rope_dim], apply_rotary_emb(kv[:, -self.rope_dim :], cos_tok, sin_tok)], dim=1)
        return q, kv

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
                apply_rotary_emb(o[..., -self.rope_dim :], cos_tok.unsqueeze(1), sin_tok.unsqueeze(1), inverse=True),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ attention (prefill)
    def _attention_prefill(self, x, infer_state, lw):
        T = x.shape[0]
        if self.compress_ratio:
            cos_tok, sin_tok = infer_state.position_cos_compress, infer_state.position_sin_compress
        else:
            cos_tok, sin_tok = infer_state.position_cos_sliding, infer_state.position_sin_sliding
        q, kv = self._qkv(x, cos_tok, sin_tok, lw)
        sink = lw.attn_sink_.weight
        o = x.new_empty(T, self.tp_q_heads, self.head_dim)
        b_req = infer_state.b_req_idx.tolist()
        starts = infer_state.b_q_start_loc.tolist()
        lens = infer_state.b_q_seq_len.tolist()
        for req, st, ln in zip(b_req, starts, lens):
            q_r, kv_r, x_r = q[st : st + ln], kv[st : st + ln], x[st : st + ln]
            kv_all, n_window, ncomp = self._gather_prefill(x_r, kv_r, req, lw, infer_state)
            ti = self._topk_idxs_prefill(ln, n_window, ncomp, x.device)
            o[st : st + ln] = torch_sparse_attn(q_r.unsqueeze(0), kv_all.unsqueeze(0), sink, ti, self.softmax_scale)[0]
        return self._out_proj(self._inv_rope(o, cos_tok, sin_tok), infer_state, lw)

    def _gather_prefill(self, x_r, kv_r, req, lw, infer_state):
        ln = kv_r.shape[0]
        if self.compress_ratio:
            comp, ks, ss = compressor_prefill_state(
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
            )
            self._state[req] = {"kv_hist": kv_r.detach(), "comp_kv": comp.detach(), "cstate_kv": ks, "cstate_score": ss}
            return torch.cat([kv_r, comp], dim=0), ln, comp.shape[0]
        self._state[req] = {"kv_hist": kv_r.detach()}
        return kv_r, ln, 0

    def _topk_idxs_prefill(self, seqlen, n_window, ncomp, device):
        t = torch.arange(seqlen, device=device)
        j = torch.arange(n_window, device=device)
        win = torch.where(
            (j.unsqueeze(0) <= t.unsqueeze(1)) & (j.unsqueeze(0) > (t.unsqueeze(1) - self.window)),
            j.unsqueeze(0).expand(seqlen, n_window),
            torch.full((seqlen, n_window), -1, device=device, dtype=torch.long),
        )
        if ncomp:
            c = torch.arange(ncomp, device=device)
            comp = torch.where(
                c.unsqueeze(0) < ((t.unsqueeze(1) + 1) // self.compress_ratio),
                (c.unsqueeze(0) + n_window).expand(seqlen, ncomp),
                torch.full((seqlen, ncomp), -1, device=device, dtype=torch.long),
            )
            return torch.cat([win, comp], dim=1).int().unsqueeze(0)
        return win.int().unsqueeze(0)

    # ------------------------------------------------------------------ attention (decode)
    def _attention_decode(self, x, infer_state, lw):
        B = x.shape[0]  # one new token per request
        if self.compress_ratio:
            cos_tok, sin_tok = infer_state.position_cos_compress, infer_state.position_sin_compress
        else:
            cos_tok, sin_tok = infer_state.position_cos_sliding, infer_state.position_sin_sliding
        q, kv = self._qkv(x, cos_tok, sin_tok, lw)  # [B, heads, hd], [B, hd]
        sink = lw.attn_sink_.weight
        b_req = infer_state.b_req_idx.tolist()
        seqlens = infer_state.b_seq_len.tolist()
        o = x.new_empty(B, self.tp_q_heads, self.head_dim)
        for i, (req, seq) in enumerate(zip(b_req, seqlens)):
            stt = self._state[req]
            stt["kv_hist"] = torch.cat([stt["kv_hist"], kv[i : i + 1]], dim=0)
            start_pos = seq - 1
            if self.compress_ratio:
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
                )
                if e is not None:
                    stt["comp_kv"] = torch.cat([stt["comp_kv"], e.unsqueeze(0)], dim=0)
                win_kv = stt["kv_hist"][-self.window :]
                kv_all = torch.cat([win_kv, stt["comp_kv"]], dim=0)
            else:
                win_kv = stt["kv_hist"][-self.window :]
                kv_all = win_kv
            ti = torch.arange(kv_all.shape[0], device=x.device).view(1, 1, -1).int()
            o[i] = torch_sparse_attn(
                q[i].view(1, 1, self.tp_q_heads, self.head_dim), kv_all.unsqueeze(0), sink, ti, self.softmax_scale
            )[0, 0]
        return self._out_proj(self._inv_rope(o, cos_tok, sin_tok), infer_state, lw)

    # ------------------------------------------------------------------ moe
    def _fp4_experts(self, x, weights, indices, lw):
        experts = lw.experts_
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        counts = torch.bincount(indices.reshape(-1), minlength=experts.n_routed_experts)
        for expert_id in torch.nonzero(counts, as_tuple=False).flatten().tolist():
            token_idx, top_idx = torch.where(indices == expert_id)
            if token_idx.numel() == 0:
                continue
            x_i = x[token_idx]
            w1 = dequant_fp4_group_to_bf16(experts.w1[expert_id], experts.w1_scale[expert_id])
            w3 = dequant_fp4_group_to_bf16(experts.w3[expert_id], experts.w3_scale[expert_id])
            gate = F.linear(x_i, w1).float().clamp(max=self.swiglu_limit)
            up = F.linear(x_i, w3).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            hidden = F.silu(gate) * up
            hidden.mul_(weights[token_idx, top_idx].unsqueeze(-1))
            w2 = dequant_fp4_group_to_bf16(experts.w2[expert_id], experts.w2_scale[expert_id])
            out.index_add_(0, token_idx, F.linear(hidden.to(x.dtype), w2).float())
        return out.to(x.dtype)

    def _moe_ffn(self, x, infer_state, lw):
        gw = lw.gate_weight_.mm_param.weight
        scores = F.softplus(F.linear(x.float(), gw.float())).sqrt()  # sqrtsoftplus
        if self.is_hash:
            indices = lw.gate_tid2eid_.weight[infer_state.input_ids.long()]
        else:
            indices = (scores + lw.gate_bias_.weight.unsqueeze(0)).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = (weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.route_scale).to(torch.float32)
        routed = self._fp4_experts(x, weights, indices.long(), lw)
        g = lw.shared_gate_.mm(x).float().clamp(max=self.swiglu_limit)
        u = lw.shared_up_.mm(x).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        shared = lw.shared_down_.mm((F.silu(g) * u).to(x.dtype))
        if self.enable_ep_moe and getattr(lw.experts_, "is_ep", False):
            if self.tp_world_size_ > 1:
                all_reduce(shared, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
            return routed + shared
        out = routed + shared
        if self.tp_world_size_ > 1:
            all_reduce(out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        return out
