import math
import torch
import torch.nn as nn

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.gemma4.layer_weights.transformer_layer_weight import Gemma4TransformerLayerWeight
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import (
    context_attention_fwd_gemma4_mm,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


class Gemma4TransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    Gemma-4 decoder block. Per-layer heterogeneity (sliding vs full attention)
    is handled by switching shape / RoPE table / sliding-window flag at init
    time. The KV cache layout is uniform (sliding shape: num_kv_heads=16,
    head_dim=256); full-attention layers pack their (4, 512) tensor into the
    first 8 heads of the 16-head slot at cache-write time, then reshape on
    read. See Gemma4TpPartModel._init_mem_manager for context.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.eps_ = network_config.get("rms_norm_eps", 1e-6)
        self.embed_dim_ = network_config["hidden_size"]
        self.is_moe = bool(network_config.get("enable_moe_block", False))
        self.num_experts_per_tok = network_config.get("num_experts_per_tok", network_config.get("top_k_experts", 0))
        self.norm_topk_prob = network_config.get("norm_topk_prob", True)
        self.router_root_scale = self.embed_dim_ ** -0.5

        layer_type = network_config["layer_types"][layer_num]
        self.is_sliding = layer_type == "sliding_attention"

        # Some E-series checkpoints leave num_global_key_value_heads = null;
        # HF treats that as "fall back to num_key_value_heads".
        num_global_kv = network_config.get("num_global_key_value_heads") or network_config["num_key_value_heads"]

        # Override parent's head_dim_ (hidden_size/num_heads = 224 on 31B, wrong
        # for Gemma-4 — actual is 256 sliding / 512 full).
        if self.is_sliding:
            self.head_dim_ = network_config["head_dim"]
            total_kv_heads = network_config["num_key_value_heads"]
            self.k_eq_v = False
        else:
            self.head_dim_ = network_config["global_head_dim"]
            total_kv_heads = num_global_kv
            self.k_eq_v = network_config.get("attention_k_eq_v", True)

        # TP shard counts for this layer
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(total_kv_heads // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_

        self.kv_cache_slot_dim_ = network_config["head_dim"]
        sliding_total = network_config["num_key_value_heads"] * network_config["head_dim"]
        full_total = num_global_kv * network_config["global_head_dim"]
        per_token_k_width = max(sliding_total, full_total)
        assert (
            per_token_k_width % self.kv_cache_slot_dim_ == 0
        ), f"per-token K width {per_token_k_width} not aligned to kv_cache_slot_dim {self.kv_cache_slot_dim_}"
        self.kv_cache_slot_num_ = (per_token_k_width // self.kv_cache_slot_dim_) // self.tp_world_size_

        # Sliding window (None on full-attn layers)
        if self.is_sliding:
            sw = network_config.get("sliding_window", 0)
            self.sliding_window_ = int(sw) if sw else 0
        else:
            self.sliding_window_ = 0

        # E-series Per-Layer Embeddings gate (HF: config.hidden_size_per_layer_input,
        # absent or 0 on 31B).
        self.has_ple_ = bool(network_config.get("hidden_size_per_layer_input"))
        if self.has_ple_:
            self.ple_dim_ = network_config["hidden_size_per_layer_input"]

        # HF: config.num_kv_shared_layers (may be missing or null on non-E
        # checkpoints — treat as 0).
        kv_shared_count = network_config.get("num_kv_shared_layers") or 0
        total_layers = network_config["num_hidden_layers"]
        self.is_kv_shared_ = kv_shared_count > 0 and layer_num >= total_layers - kv_shared_count
        self.kv_share_target_layer_ = None
        if self.is_kv_shared_:
            cutoff = total_layers - kv_shared_count
            for j in range(cutoff - 1, -1, -1):
                if network_config["layer_types"][j] == layer_type:
                    self.kv_share_target_layer_ = j
                    break
            assert self.kv_share_target_layer_ is not None, (
                f"layer {layer_num} ({layer_type}) is KV-shared but no earlier non-shared "
                f"layer of the same type found below cutoff={cutoff}"
            )

        # Always 1.0: NoPE dims for full-attn layers are zero-padded into
        # cos/sin (cos=1, sin=0 → identity), so the kernel walks the whole
        # head_dim. Don't change to 0.25 — that double-counts with the table.
        self.partial_rotary_factor_ = 1.0

        self.ple_static_buffer = None

    def _rope_cos_sin(self, infer_state):
        # Tables are built in the model dtype (Gemma4TpPartModel._init_to_get_rotary_gemma4),
        # so they already match q/k dtype — no cast needed.
        if self.is_sliding:
            return infer_state.position_cos_sliding, infer_state.position_sin_sliding
        return infer_state.position_cos_full, infer_state.position_sin_full

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight) -> torch.Tensor:
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        head_dim = self.head_dim_
        q_heads = self.tp_q_head_num_
        kv_heads = self.tp_k_head_num_

        q = layer_weight.q_proj.mm(input).view(-1, q_heads, head_dim)
        q = layer_weight.q_norm_weight_(input=q, eps=self.eps_, alloc_func=self.alloc_tensor)

        cos, sin = self._rope_cos_sin(infer_state)

        if self.is_kv_shared_:
            # K/V come from target layer's already-rotated, already-normed cache.
            rotary_emb_fwd(q, None, cos, sin, partial_rotary_factor=self.partial_rotary_factor_)
            q = q * math.sqrt(head_dim)
            if infer_state.need_dp_prefill_balance:
                q = infer_state._all_to_all_unbalance_get(data=q)
            return q, None

        # ---- non-shared: full K/V path ----
        k = layer_weight.k_proj.mm(input).view(-1, kv_heads, head_dim)
        if self.k_eq_v:
            # Full-attn k_eq_v variant (e.g. 31B): K weights serve as V.
            v = k
        else:
            v = layer_weight.v_proj.mm(input).view(-1, kv_heads, head_dim)

        k = layer_weight.k_norm_weight_(input=k, eps=self.eps_, alloc_func=self.alloc_tensor)

        # V-norm: unweighted RMSNorm over head_dim (matches vllm's Gemma4 has_weight=False).
        v = rmsnorm_forward(
            x=v,
            weight=None,
            eps=self.eps_,
            out=self.alloc_tensor(v.shape, dtype=v.dtype, device=v.device),
        )

        rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=self.partial_rotary_factor_)

        # Gemma-4 uses scaling=1.0 in attention. The attention kernel hardcodes
        # sm_scale = 1/sqrt(head_dim); pre-scale Q by sqrt(head_dim) so the
        # kernel's division cancels out, yielding scores = Q @ K^T.
        q = q * math.sqrt(head_dim)

        # Pack into the uniform KV-cache layout (N, 2*slot_num, slot_dim).
        # K occupies slots [0, used_slots); V occupies
        # [slot_num, slot_num + used_slots). If this layer's K/V width is
        # smaller than the allocated cache slot width, pad with zeros.
        cache_slot_num = self.kv_cache_slot_num_
        cache_slot_dim = self.kv_cache_slot_dim_
        N = k.shape[0]
        k_packed = k.reshape(N, -1, cache_slot_dim)
        v_packed = v.reshape(N, -1, cache_slot_dim)
        used_cache_slots = k_packed.shape[1]
        if used_cache_slots == cache_slot_num:
            cache_kv = torch.cat([k_packed, v_packed], dim=1)
        else:
            cache_kv = self.alloc_tensor((N, 2 * cache_slot_num, cache_slot_dim), dtype=k.dtype)
            cache_kv.zero_()
            cache_kv[:, :used_cache_slots, :] = k_packed
            cache_kv[:, cache_slot_num : cache_slot_num + used_cache_slots, :] = v_packed

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    def _post_cache_kv(self, cache_kv, infer_state, layer_weight):
        if self.is_kv_shared_ or cache_kv is None:
            return
        return super()._post_cache_kv(cache_kv, infer_state, layer_weight)

    # ----- Attention kernels (sliding window + per-layer KV reshape) ---

    def _att_control(self):
        if self.is_sliding and self.sliding_window_ > 0:
            w = self.sliding_window_ - 1
            return AttControl(use_sliding_window=True, sliding_window=(w, 0))
        return AttControl(use_sliding_window=False, sliding_window=(-1, -1))

    def _get_layer_kv(self, infer_state: InferStateInfo):
        # KV-shared layers read from the target layer's cache slot.
        layer_idx = self.kv_share_target_layer_ if self.is_kv_shared_ else self.layer_num_
        _k_raw, _v_raw = infer_state.mem_manager.get_att_input_params(layer_index=layer_idx)
        # _k_raw / _v_raw shape (S, cache_slot_num, cache_slot_dim). Use .view
        # (not .reshape) so any non-contiguous layout from a future mem_manager
        # backend fails loudly instead of silently copying — slice + view is
        # O(1) on the standard MemoryManager layout (inner (kv_heads, head_dim)
        # span is contiguous).
        kv_heads = self.tp_k_head_num_
        head_dim = self.head_dim_
        cache_slot_dim = self.kv_cache_slot_dim_
        used_cache_slots = kv_heads * head_dim // cache_slot_dim
        if used_cache_slots == _k_raw.shape[1]:
            # Layout already matches this layer's natural shape.
            return _k_raw.view(-1, kv_heads, head_dim), _v_raw.view(-1, kv_heads, head_dim)
        # Otherwise the K/V live in the first used_cache_slots; the rest is zero pad.
        _k = _k_raw[:, :used_cache_slots, :].view(-1, kv_heads, head_dim)
        _v = _v_raw[:, :used_cache_slots, :].view(-1, kv_heads, head_dim)
        return _k, _v

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv,
        infer_state: InferStateInfo,
        layer_weight: Gemma4TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        _k, _v = self._get_layer_kv(infer_state)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        if self.is_sliding:
            # Sliding layers always go through the gemma4_mm Triton kernel: it
            # handles SWA + image bidirectional masking in one pass.
            o_tensor = self.alloc_tensor(_q.shape, q.dtype)
            sw = (self.sliding_window_ - 1, 0) if self.sliding_window_ > 0 else (-1, -1)
            context_attention_fwd_gemma4_mm(
                _q,
                _k,
                _v,
                o_tensor,
                infer_state.b_req_idx,
                infer_state.b_q_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_q_seq_len,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_image_token_end,
                sliding_window=sw,
            )
            return o_tensor.view(q.shape)

        # Full-attn layers: head_dim=512, no SWA, no image bidi — standard
        # triton via backend1.
        o_tensor = infer_state.prefill_att_state1.prefill_att(
            q=_q, k=_k, v=_v, att_control=self._att_control(), alloc_func=self.alloc_tensor
        )
        return o_tensor.view(q.shape)

    def _token_attention_kernel(
        self,
        q: torch.Tensor,
        infer_state: InferStateInfo,
        layer_weight: Gemma4TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        _k, _v = self._get_layer_kv(infer_state)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        att_state = infer_state.decode_att_state if self.is_sliding else infer_state.decode_att_state1
        o_tensor = att_state.decode_att(q=_q, k=_k, v=_v, att_control=self._att_control(), alloc_func=self.alloc_tensor)
        return o_tensor.view(q.shape)

    # ----- FFN (Gemma gelu-tanh, fused gate_up + down) -----------------

    def _ffn_dense(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        gate_up = layer_weight.gate_up_proj.mm(input)
        ffn1 = self.alloc_tensor((input.size(0), gate_up.size(1) // 2), input.dtype)
        silu_and_mul_fwd(gate_up, ffn1)
        gate_up = None
        ffn2 = layer_weight.down_proj.mm(ffn1)
        ffn1 = None
        ffn2 = self._tpsp_reduce(input=ffn2, infer_state=infer_state)
        return ffn2

    def _router_logits(self, residual, layer_weight: Gemma4TransformerLayerWeight) -> torch.Tensor:
        # Mirrors vllm Gemma4Router: unweighted RMSNorm -> 1/sqrt(hidden) ->
        # per-channel scale -> bf16xbf16 -> fp32 gate matmul for stable top-k.
        x = residual.view(-1, self.embed_dim_)
        x = rmsnorm_forward(x=x, weight=None, eps=self.eps_, out=self.alloc_tensor(x.shape, dtype=x.dtype))
        x = x * self.router_root_scale * layer_weight.router_input_scale_.weight
        return layer_weight.moe_gate.mm(x.to(torch.float32))

    def _ffn_moe(self, input, router_logits, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight):
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        moe_out = layer_weight.experts.experts(
            input,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
        )
        moe_out = self._tpsp_reduce(input=moe_out, infer_state=infer_state)
        return moe_out

    def _ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight):
        residual = input_embdings
        dense_input = layer_weight.pre_feedforward_layernorm_weight_(
            input=residual, eps=self.eps_, alloc_func=self.alloc_tensor
        )
        dense_out = self._ffn_dense(dense_input, infer_state, layer_weight)
        dense_input = None

        if self.is_moe:
            dense_out = layer_weight.post_feedforward_layernorm_1_weight_(
                input=dense_out, eps=self.eps_, alloc_func=self.alloc_tensor
            )

            router_logits = self._router_logits(residual, layer_weight)
            moe_input = layer_weight.pre_feedforward_layernorm_2_weight_(
                input=residual, eps=self.eps_, alloc_func=self.alloc_tensor
            )
            moe_out = self._ffn_moe(moe_input, router_logits, infer_state, layer_weight)
            moe_input = None
            router_logits = None
            moe_out = layer_weight.post_feedforward_layernorm_2_weight_(
                input=moe_out, eps=self.eps_, alloc_func=self.alloc_tensor
            )
            dense_out.add_(moe_out)
            moe_out = None

        ffn_out = layer_weight.post_feedforward_layernorm_weight_(
            input=dense_out, eps=self.eps_, alloc_func=self.alloc_tensor
        )
        dense_out = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    # ----- block-level forwards (PLE fusion + layer_scalar at the end) ----

    def _block_epilogue(self, hidden_states, infer_state, layer_weight):
        if self.has_ple_:
            flat = hidden_states.view(-1, self.embed_dim_)
            N = flat.shape[0]
            ple_slice = self.ple_static_buffer[:N, self.layer_num_, :]
            gate = layer_weight.per_layer_input_gate_.mm(flat)
            gated = nn.functional.gelu(gate, approximate="tanh") * ple_slice
            contrib = layer_weight.per_layer_projection_.mm(gated)
            contrib = layer_weight.post_per_layer_input_norm_weight_(
                input=contrib, eps=self.eps_, alloc_func=self.alloc_tensor
            )
            flat.add_(contrib)
        hidden_states.mul_(layer_weight.layer_scalar_.weight)
        return hidden_states

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight):
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_), infer_state, layer_weight)
        o = self.context_attention_forward(input1, infer_state, layer_weight)
        input1 = None
        # Gemma sandwich norm: post_attention_layernorm on the attn branch
        # before the residual add, not on the post-add residual stream.
        o = self._ffn_norm(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input_embdings = self._ffn(input_embdings, infer_state, layer_weight)

        return self._block_epilogue(input_embdings, infer_state, layer_weight)

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight):
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_), infer_state, layer_weight)
        o = self.token_attention_forward(input1, infer_state, layer_weight)
        input1 = None
        o = self._ffn_norm(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input_embdings = self._ffn(input_embdings, infer_state, layer_weight)

        return self._block_epilogue(input_embdings, infer_state, layer_weight)
