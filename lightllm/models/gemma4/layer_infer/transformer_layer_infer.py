import math
import torch
import torch.nn as nn

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.gemma4.layer_weights.transformer_layer_weight import Gemma4TransformerLayerWeight
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
        self.eps_ = 1e-6
        self.embed_dim_ = network_config["hidden_size"]

        layer_type = network_config["layer_types"][layer_num]
        self.is_sliding = layer_type == "sliding_attention"

        if self.is_sliding:
            self.layer_head_dim_ = network_config["head_dim"]
            total_kv_heads = network_config["num_key_value_heads"]
            self.k_eq_v = False
        else:
            self.layer_head_dim_ = network_config["global_head_dim"]
            total_kv_heads = network_config["num_global_key_value_heads"]
            self.k_eq_v = network_config.get("attention_k_eq_v", True)

        # TP shard counts for this layer
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(total_kv_heads // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_

        # Uniform mem-manager layout (sliding shape per rank)
        self.mm_head_dim_ = network_config["head_dim"]
        self.mm_kv_head_num_ = network_config["num_key_value_heads"] // self.tp_world_size_

        # Sliding window (None on full-attn layers)
        if self.is_sliding:
            sw = network_config.get("sliding_window", 0)
            self.sliding_window_ = int(sw) if sw else 0
        else:
            self.sliding_window_ = 0

        # Partial rotary factor for the RoPE kernel. The sliding table is sized
        # (seq, head_dim/2) so full rotation over head_dim is the default.
        # The full table is sized (seq, global_head_dim/2) with zero-padded
        # frequencies (proportional RoPE) — we still pass partial_rotary_factor=1
        # to the kernel so it walks every pair, applying identity for the zeroed
        # frequencies.
        self.rotary_partial_factor_ = 1.0

    def _bind_func(self):
        # Skip LlamaTransformerLayerInfer._bind_norm (it rebinds to Llama _att_norm / _ffn_norm);
        # we want our own gemma-style norm implementations below.
        return

    # ----- norms ---------------------------------------------------------

    def _att_norm(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_(
            input=input, eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _ffn_norm(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        # NOTE: gemma packs post_attention_layernorm under `ffn_norm_weight_`
        return layer_weight.ffn_norm_weight_(
            input=input, eps=self.eps_, alloc_func=self.alloc_tensor
        )

    # ----- QKV + attention ---------------------------------------------

    def _get_qkv(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        head_dim = self.layer_head_dim_
        q_heads = self.tp_q_head_num_
        kv_heads = self.tp_k_head_num_

        q = layer_weight.q_proj.mm(input).view(-1, q_heads, head_dim)
        k = layer_weight.k_proj.mm(input).view(-1, kv_heads, head_dim)
        if self.k_eq_v:
            # Full-attn layers share K weights for V.
            v = k.clone()
        else:
            v = layer_weight.v_proj.mm(input).view(-1, kv_heads, head_dim)

        # QK RMSNorm (learnable weight, Gemma-style `(1+w)` applied in fp32).
        # Reshape to 2D (N*heads, head_dim) so NoTpGEMMANormWeight accepts it.
        q_flat = q.reshape(-1, head_dim).float()
        k_flat = k.reshape(-1, head_dim).float()
        q_flat = layer_weight.q_norm_weight_(input=q_flat, eps=self.eps_, alloc_func=self.alloc_tensor)
        k_flat = layer_weight.k_norm_weight_(input=k_flat, eps=self.eps_, alloc_func=self.alloc_tensor)
        q = q_flat.view(-1, q_heads, head_dim).to(input.dtype)
        k = k_flat.view(-1, kv_heads, head_dim).to(input.dtype)

        # V-norm: unweighted RMSNorm over head_dim (matches vllm's Gemma4 has_weight=False).
        v_fp = v.float()
        v_fp = v_fp * torch.rsqrt(v_fp.pow(2).mean(dim=-1, keepdim=True) + self.eps_)
        v = v_fp.to(input.dtype)

        # Per-layer RoPE
        if self.is_sliding:
            cos = infer_state.position_cos_sliding.to(q.dtype)
            sin = infer_state.position_sin_sliding.to(q.dtype)
        else:
            cos = infer_state.position_cos_full.to(q.dtype)
            sin = infer_state.position_sin_full.to(q.dtype)
        rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=self.rotary_partial_factor_)

        # Gemma-4 uses scaling=1.0 in attention. The attention kernel hardcodes
        # sm_scale = 1/sqrt(head_dim); pre-scale Q by sqrt(head_dim) so the
        # kernel's division cancels out, yielding scores = Q @ K^T.
        q = q * math.sqrt(head_dim)

        # Pack into the uniform mem-manager layout.
        mm_heads = self.mm_kv_head_num_
        mm_dim = self.mm_head_dim_
        if self.is_sliding:
            # (N, 2*mm_heads, mm_dim) with [:mm_heads]=K, [mm_heads:]=V
            cache_kv = torch.cat([k, v], dim=1)
        else:
            # K,V shape (N, kv_heads, layer_head_dim) e.g. (N, 2, 512) on tp=2.
            # Reshape each half to (N, kv_heads*layer_head_dim // mm_dim, mm_dim) e.g. (N, 4, 256) on tp=2.
            # The mem-manager layout has (N, 2*mm_heads, mm_dim) = (N, 16, 256) on tp=2 for this
            # checkpoint — pad to that shape with zeros on unused head slots.
            N = k.shape[0]
            k_packed = k.reshape(N, -1, mm_dim)  # (N, kv_heads * layer_head_dim // mm_dim, mm_dim)
            v_packed = v.reshape(N, -1, mm_dim)
            cache_kv = self.alloc_tensor((N, 2 * mm_heads, mm_dim), dtype=k.dtype)
            cache_kv.zero_()
            k_slots = k_packed.shape[1]
            cache_kv[:, :k_slots, :] = k_packed
            cache_kv[:, mm_heads : mm_heads + k_slots, :] = v_packed

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    def _get_o(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        input = input.view(-1, self.tp_o_head_num_ * self.layer_head_dim_)
        o_tensor = layer_weight.o_proj.mm(input)
        o_tensor = self._tpsp_reduce(input=o_tensor, infer_state=infer_state)
        return o_tensor

    # ----- Attention kernels (sliding window + per-layer KV reshape) ---

    def _att_control(self):
        # SWA is only safe with FA3 (it consumes window_size per-call). Triton
        # backend asserts use_sliding_window is False; lightllm's flashinfer
        # wrapper plans once and ignores per-call windows. The flag is set
        # by Gemma4TpPartModel._init_att_backend after backend selection.
        if self.is_sliding and self.sliding_window_ > 0 and self.network_config_.get("_gemma4_use_swa", False):
            w = self.sliding_window_ - 1
            return AttControl(use_sliding_window=True, sliding_window=(w, w))
        return AttControl(use_sliding_window=False, sliding_window=(-1, -1))

    def _get_layer_kv(self, infer_state: InferStateInfo):
        _k_raw, _v_raw = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        # _k_raw / _v_raw shape (S, mm_heads, mm_dim)
        if self.is_sliding:
            # sliding K is stored in the full (mm_heads, mm_dim) slot; head count matches.
            return _k_raw, _v_raw
        # full layer: the real K/V live in the first `kv_heads * layer_head_dim // mm_dim`
        # head slots. Reshape to (S, kv_heads, layer_head_dim).
        kv_heads = self.tp_k_head_num_
        head_dim = self.layer_head_dim_
        mm_dim = self.mm_head_dim_
        k_slots = kv_heads * head_dim // mm_dim
        _k = _k_raw[:, :k_slots, :].reshape(-1, kv_heads, head_dim)
        _v = _v_raw[:, :k_slots, :].reshape(-1, kv_heads, head_dim)
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
        _q = q.view(-1, self.tp_q_head_num_, self.layer_head_dim_)
        o_tensor = infer_state.prefill_att_state.prefill_att(
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
        _q = q.view(-1, self.tp_q_head_num_, self.layer_head_dim_)
        o_tensor = infer_state.decode_att_state.decode_att(
            q=_q, k=_k, v=_v, att_control=self._att_control(), alloc_func=self.alloc_tensor
        )
        return o_tensor.view(q.shape)

    # ----- FFN (Gemma gelu-tanh, separate gate/up/down) ----------------

    def _ffn(
        self, input, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        gate = layer_weight.gate_proj.mm(input)
        up = layer_weight.up_proj.mm(input)
        ffn1 = nn.functional.gelu(gate, approximate="tanh") * up
        gate = None
        up = None
        ffn2 = layer_weight.down_proj.mm(ffn1)
        ffn1 = None
        ffn2 = self._tpsp_reduce(input=ffn2, infer_state=infer_state)
        return ffn2

    # ----- block-level forwards (add layer_scalar at the end) ----------

    def _apply_layer_scalar(self, hidden_states, layer_weight):
        hidden_states.mul_(layer_weight.layer_scalar_.weight)
        return hidden_states

    def context_forward(
        self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ):
        input_embdings = input_embdings.to(torch.bfloat16)

        # attn sub-block
        input1 = self._att_norm(
            input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight
        ).to(torch.bfloat16)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        # ffn sub-block
        input1 = layer_weight.pre_feedforward_layernorm_weight_(
            input=input_embdings.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        ffn_out = layer_weight.post_feedforward_layernorm_weight_(
            input=ffn_out.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

        return self._apply_layer_scalar(input_embdings, layer_weight)

    def token_forward(
        self, input_embdings, infer_state: InferStateInfo, layer_weight: Gemma4TransformerLayerWeight
    ):
        input_embdings = input_embdings.to(torch.bfloat16)

        input1 = self._att_norm(
            input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight
        ).to(torch.bfloat16)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = layer_weight.pre_feedforward_layernorm_weight_(
            input=input_embdings.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        ffn_out = layer_weight.post_feedforward_layernorm_weight_(
            input=ffn_out.float(), eps=self.eps_, alloc_func=self.alloc_tensor
        ).to(torch.bfloat16)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

        return self._apply_layer_scalar(input_embdings, layer_weight)
