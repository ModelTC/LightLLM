"""Baseline BF16 DSA arithmetic with the main-branch attention state API."""

import os

import torch
import torch.nn.functional as F

from .dsa import project_indexer
from .quick_rope import apply_partial_mrope
from .triton_kernel.dsa import paged_dsa_attention


# ``auto`` routes only the regime where Indexer/TopK cannot amortize its
# fixed launch cost.  The old values remain available for controlled A/B
# runs: ``1`` means the original ``score_width <= topk`` rule and ``0``
# disables this router.
SHORT_DENSE_MODE = os.environ.get("DSA_SHORT_DENSE_FA3", "auto").lower()
SHORT_DENSE_PREFILL_MAX_KEYS = int(os.environ.get("DSA_SHORT_DENSE_PREFILL_MAX_KEYS", "131072"))
SHORT_DENSE_DECODE_MAX_KEYS = int(os.environ.get("DSA_SHORT_DENSE_DECODE_MAX_KEYS", "131072"))
SHORT_DENSE_DECODE_MAX_ROWS = int(os.environ.get("DSA_SHORT_DENSE_DECODE_MAX_ROWS", "1"))
PREFILL_POLICY = os.environ.get("DSA_PREFILL_POLICY", "dsa").lower()
PREFILL_DENSE_Q_HEADS_MAX = int(os.environ.get("DSA_PREFILL_DENSE_Q_HEADS_MAX", "8"))
if (
    SHORT_DENSE_MODE not in ("0", "1", "auto", "off", "on")
    or SHORT_DENSE_PREFILL_MAX_KEYS < 1
    or SHORT_DENSE_DECODE_MAX_KEYS < 1
    or SHORT_DENSE_DECODE_MAX_ROWS < 1
    or PREFILL_POLICY not in ("dsa", "dense", "auto")
    or PREFILL_DENSE_Q_HEADS_MAX < 1
):
    raise ValueError("Invalid DSA prefill policy")


class DsaAttentionMixin:
    def _use_short_dense_attention(self, infer_state, rows):
        """Route short P or tiny single-request D to the native FA3 path."""
        if SHORT_DENSE_MODE in ("0", "off"):
            return False
        if SHORT_DENSE_MODE in ("1", "on"):
            return infer_state.dsa_score_max_keys <= self.dsa["topk"]
        if infer_state.is_prefill:
            return infer_state.dsa_score_max_keys <= SHORT_DENSE_PREFILL_MAX_KEYS
        return (
            rows <= SHORT_DENSE_DECODE_MAX_ROWS
            and infer_state.dsa_score_max_keys <= SHORT_DENSE_DECODE_MAX_KEYS
        )

    def _use_dense_prefill(self, infer_state):
        if not infer_state.is_prefill:
            return False
        if PREFILL_POLICY == "dense":
            return True
        if PREFILL_POLICY == "dsa":
            return False
        return (
            self.tp_q_head_num_ <= PREFILL_DENSE_Q_HEADS_MAX
            and infer_state.dsa_score_max_keys > self.dsa["topk"]
        )
    def _get_qkv(self, input, infer_state, layer_weight):
        if not self.use_dsa:
            return super()._get_qkv(input, infer_state, layer_weight)
        input = self._tpsp_allgather(input.view(-1, self.embed_dim_), infer_state)
        projected = layer_weight.qkvo_gate_proj.mm(input)
        qsize = self.tp_q_head_num_ * self.head_dim_
        kvsize = (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_
        qkv, gate = projected.split([qsize + kvsize, qsize], dim=-1)
        q, kv = qkv.split([qsize, kvsize], dim=-1)
        infer_state.gate_logics_value = gate
        layer_weight.qk_norm_weight_(q, kv[:, : self.tp_k_head_num_ * self.head_dim_], eps=self.eps_)
        kv = kv.view(-1, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        apply_partial_mrope(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, : self.tp_k_head_num_],
            infer_state.position_cos,
            infer_state.position_sin,
            self.dsa["rotary_dim"],
            self.dsa_mrope_sections,
        )
        dense_prefill = self._use_dense_prefill(infer_state)
        short = dense_prefill or self._use_short_dense_attention(infer_state, input.shape[0])
        if short:
            ik = F.layer_norm(
                F.linear(input, layer_weight.dsa_wk.weight),
                (self.dsa["head_dim"],),
                layer_weight.dsa_norm.weight,
                layer_weight.dsa_norm.bias,
                1e-6,
            )
            iq, iw = None, None
        else:
            iq, ik, iw = project_indexer(
                input,
                layer_weight.dsa_wq.weight,
                layer_weight.dsa_wk.weight,
                layer_weight.dsa_weights.weight,
                layer_weight.dsa_norm.weight,
                layer_weight.dsa_norm.bias,
                self.dsa["num_heads"],
            )
        apply_partial_mrope(
            iq,
            ik[:, None],
            infer_state.position_cos,
            infer_state.position_sin,
            self.dsa["rotary_dim"],
            self.dsa_mrope_sections,
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            kv = infer_state._all_to_all_unbalance_get(data=kv)
            ik = infer_state._all_to_all_unbalance_get(data=ik)
            if iq is not None:
                iq = infer_state._all_to_all_unbalance_get(data=iq)
                iw = infer_state._all_to_all_unbalance_get(data=iw)
        indexer_state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role,
            state_type_filter="dsa_indexer",
            logical_layer_id=self.layer_num_,
        )
        indexer_state.write_full_att_kv(self.layer_num_, infer_state.mem_index, ik[:, None])
        # Carry Q and weights through prefill graph callbacks, as in baseline.
        if not short:
            q = torch.cat((q, iq.flatten(1), iw), dim=-1)
        return q, kv

    def _post_cache_kv(self, cache_kv, infer_state, layer_weight):
        if not self.use_dsa:
            return super()._post_cache_kv(cache_kv, infer_state, layer_weight)
        # This execution layer owns both GQA and Indexer resources.
        state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role,
            state_type_filter="gqa",
            logical_layer_id=self.layer_num_,
        )
        state.write_full_att_kv(self.layer_num_, infer_state.mem_index, cache_kv)

    def _dsa_attention(self, packed_q, infer_state):
        qdim = self.tp_q_head_num_ * self.head_dim_
        idim = self.dsa["num_heads"] * self.dsa["head_dim"]
        q, iq, iw = packed_q.split([qdim, idim, self.dsa["num_heads"]], dim=-1)
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        iq = iq.view(-1, self.dsa["num_heads"], self.dsa["head_dim"])
        manager, role = infer_state.att_state_manager, infer_state.att_state_role
        key, value = manager.get_state(role, "gqa", self.layer_num_).get_layer_view(self.layer_num_)
        ik = manager.get_state(role, "dsa_indexer", self.layer_num_).get_layer_view(self.layer_num_)
        return paged_dsa_attention(
            q,
            iq,
            iw,
            key,
            value,
            ik,
            infer_state,
            self.dsa["topk"],
            infer_state.dsa_score_max_keys,
        ).flatten(1)

    def _context_attention_kernel(self, q, kv, infer_state, layer_weight):
        if not self.use_dsa:
            return super()._context_attention_kernel(q, kv, infer_state, layer_weight)
        if self._use_dense_prefill(infer_state) or self._use_short_dense_attention(infer_state, q.shape[0]):
            return super()._context_attention_kernel(q, kv, infer_state, layer_weight)
        return self._dsa_attention(q, infer_state)

    def _token_attention_kernel(self, q, infer_state, layer_weight):
        if not self.use_dsa:
            return super()._token_attention_kernel(q, infer_state, layer_weight)
        if self._use_short_dense_attention(infer_state, q.shape[0]):
            return super()._token_attention_kernel(q, infer_state, layer_weight)
        return self._dsa_attention(q, infer_state)
