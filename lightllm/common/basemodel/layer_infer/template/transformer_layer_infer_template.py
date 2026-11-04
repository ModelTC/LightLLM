from lightllm.common.attention_state.full_att import FullAttentionState
import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from lightllm.distributed import all_reduce
from typing import Tuple


class TransformerLayerInferTpl(TransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        # need to set by subclass
        self.eps_ = 1e-5
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        self.head_dim_ = -1
        self.embed_dim_ = -1
        return

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _post_cache_kv(self, cache_kv, infer_state: InferStateInfo, layer_weight):
        if getattr(self, "bound_kv_writer", None) is not None:
            self.bound_kv_writer(token_indices=infer_state.mem_index, kv=cache_kv)
            return
        state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role,
            state_type_filter="",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )
        state.write_full_att_kv(
            logical_layer_id=self.layer_num_,
            token_indices=infer_state.mem_index,
            kv=cache_kv,
        )
        return

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def context_attention_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        q, cache_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )
        q = None
        o = self._get_o(o, infer_state, layer_weight)

        return o

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        o = self.context_attention_forward(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None

        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_attention_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        q, cache_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)

        return o

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        o = self.token_attention_forward(input1, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)

        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def _context_attention_extra_inputs(self, infer_state):
        return ()

    def _context_attention_output_specs(self, q):
        return None

    def _context_attention_wrapper_run(
        self, q: torch.Tensor, cache_kv: torch.Tensor, infer_state: InferStateInfo, layer_weight
    ) -> torch.Tensor:
        program = infer_state.prefill_program
        if program is None or not program.is_capturing:
            return self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        inputs = (q.contiguous(), cache_kv.contiguous() if cache_kv is not None else None)
        inputs += self._context_attention_extra_inputs(infer_state)

        def attention(tensors, batch):
            query, kv, *extra = tensors
            kwargs = {"extra_inputs": tuple(extra)} if extra else {}
            return (self._context_attention_kernel(query, kv, batch, layer_weight, **kwargs),)

        return program.call(attention, inputs, infer_state, self._context_attention_output_specs(q))[0]
