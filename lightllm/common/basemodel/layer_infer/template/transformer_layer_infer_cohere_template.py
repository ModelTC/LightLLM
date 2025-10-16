from functools import partial
from typing import Tuple

import torch
import torch.distributed as dist

from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time

from ...infer_struct import InferStateInfo
from ..transformer_layer_infer import TransformerLayerInfer
from lightllm.distributed.communication_op import all_reduce


class TransformerLayerCohereInferTpl(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, network_config, mode):
        super().__init__(layer_num, network_config, mode)

        self.use_qk_norm_ = self.network_config_.get("use_qk_norm", False)
        return

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _q_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _k_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _bind_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        self._att_norm = partial(TransformerLayerCohereInferTpl._q_norm, self)
        self._q_norm = partial(TransformerLayerCohereInferTpl._k_norm, self)
        self._k_norm = partial(TransformerLayerCohereInferTpl._att_norm, self)

    def _rotary_emb_fwd(self, q, kv, position_cos, position_sin):
        raise Exception("need to impl")

    def _bind_rotary_emb_fwd(self):
        raise Exception("need to impl")

    def _get_qkv(
        self, input, infer_state: InferStateInfo, layer_weight
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        if self.use_qk_norm_:
            q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
            k = cache_kv[:, 0 : self.tp_k_head_num_, :]
            q = self._q_norm(q, infer_state, layer_weight)
            cache_kv[:, 0 : self.tp_k_head_num_, :] = self._k_norm(k, infer_state, layer_weight)
        self._rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _context_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        q, cache_kv = self._get_qkv(input_embding, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._attn_out = o
        return

    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        ffn_out = self._ffn(input_embdings, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._ffn_out = ffn_out
        return

    def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        q, cache_kv = self._get_qkv(input_embding, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._attn_out = o
        return

    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        ffn_out = self._ffn(input_embdings, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, group=infer_state.dist_group, op=dist.ReduceOp.SUM, async_op=False)
        infer_state._ffn_out = ffn_out
        return

    def _cohere_residual(self, input_embdings, infer_state: InferStateInfo):
        # emb_addr = input_embdings.data_ptr()
        # attn_out_addr = infer_state._attn_out.data_ptr()
        # ffn_addr = infer_state._ffn_out.data_ptr()
        # assert emb_addr != attn_out_addr
        # assert emb_addr != ffn_addr
        # assert attn_out_addr != ffn_addr
        input_embdings.add_(
            infer_state._attn_out.view(-1, self.embed_dim_) + infer_state._ffn_out.view(-1, self.embed_dim_)
        )

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        self._context_attention(input1, infer_state, layer_weight=layer_weight)
        self._context_ffn(input1, infer_state, layer_weight)
        self._cohere_residual(input_embdings, infer_state)
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        self._token_attention(input1, infer_state, layer_weight=layer_weight)
        self._token_ffn(input1, infer_state, layer_weight)
        self._cohere_residual(input_embdings, infer_state)
        return input_embdings
