from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs, gqa_kv_quant_fields
from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.basemodel.attention_backend_manager import LayerAttentionConfig, attention_is_causal
import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from lightllm.distributed import all_reduce
from typing import Tuple
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class TransformerLayerInferTpl(TransformerLayerInfer):
    """ """

    # Bound by the model once its cache is allocated, from attention_config(model).
    prefill_backend = None
    decode_backend = None

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

    def attention_config(self, model) -> LayerAttentionConfig:
        """Describe this layer's attention once the model's cache has been allocated.

        The default is GQA over the layer's full-attention KV cache, the same in prefill and decode.
        """
        config = self._gqa_config(model)
        return LayerAttentionConfig(prefill=config, decode=config)

    def _gqa_cache_state(self, manager, model_role) -> FullAttentionState:
        return manager.get_state(
            model_role=model_role,
            state_type_filter="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )

    def _gqa_kv_cache(self, state: FullAttentionState):
        """This layer's (keys, values) cache views; KV-sharing layers read another layer's."""
        return state.get_layer_view(logical_layer_id=self.layer_num_)

    def _gqa_config(self, model, **fields) -> GQALayerConfig:
        """The layer's GQA config; fields add the model's own attributes (sinks, ALiBi, windows)."""
        state = self._gqa_cache_state(model.att_state_manager, model.att_state_role)
        keys, _ = self._gqa_kv_cache(state)
        scales = state.get_layer_kv_scales(self.layer_num_)
        return GQALayerConfig(
            num_heads=self.tp_q_head_num_,
            num_kv_heads=keys.shape[1],
            head_dim=self.head_dim_,
            dtype=model.data_type,
            causal=attention_is_causal(model),
            **gqa_kv_quant_fields(model.args.llm_kv_type, scales, self.head_dim_),
            **fields,
        )

    def _gqa_sinks(self, layer_weight):
        return None

    def _gqa_alibi_slopes(self, layer_weight):
        return None

    def _gqa_attention(self, backend, q, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        state = self._gqa_cache_state(infer_state.att_state_manager, infer_state.att_state_role)
        keys, values = self._gqa_kv_cache(state)
        config = backend.layer_config
        query = q.view(-1, config.num_heads, config.head_dim)
        inputs = GQALayerInputs(
            q=query,
            k_cache=keys,
            v_cache=values,
            kv_scales=state.get_layer_kv_scales(self.layer_num_),
            sinks=self._gqa_sinks(layer_weight),
            alibi_slopes=self._gqa_alibi_slopes(layer_weight),
        )
        output = backend.forward(
            infer_state.att_states[backend.state_key],
            inputs,
            out=self.alloc_tensor(query.shape, dtype=query.dtype, device=query.device),
            alloc_func=self.alloc_tensor,
        )
        return output.view(q.shape)

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _post_cache_kv(self, cache_kv, infer_state: InferStateInfo, layer_weight):
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
        return self._gqa_attention(self.prefill_backend, q, infer_state, layer_weight)

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(self.decode_backend, q, infer_state, layer_weight)

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

    def _context_attention_wrapper_run(
        self, q: torch.Tensor, cache_kv: torch.Tensor, infer_state: InferStateInfo, layer_weight
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            q = q.contiguous()
            # cache_kv is None for layers that own no K/V slot (e.g. gemma4
            # KV-shared layers, which read K/V from a prior layer's cache and
            # ignore this arg in _context_attention_kernel). Skip the
            # graph-input plumbing for it instead of crashing on None.
            cache_kv = cache_kv.contiguous() if cache_kv is not None else None
            _q = tensor_to_no_ref_tensor(q)
            _cache_kv = tensor_to_no_ref_tensor(cache_kv) if cache_kv is not None else None
            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            def get_o_shape_dtype_device():
                # 在一个新的 graph 中尝试运行，并不是为了捕获图，是为了尝试得到 o 的形状等信息
                with torch.cuda.graph(cuda_graph=torch.cuda.CUDAGraph()):
                    __o = self._context_attention_kernel(_q, _cache_kv, infer_state, layer_weight)
                    o_shape = __o.shape
                    o_dtype = __o.dtype
                    o_device = __o.device
                    del __o

                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                return o_shape, o_dtype, o_device

            o_shape, o_dtype, o_device = get_o_shape_dtype_device()
            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            o = torch.empty(o_shape, dtype=o_dtype, device=o_device)
            _o = tensor_to_no_ref_tensor(o)

            def att_func(new_infer_state: InferStateInfo):
                tmp_o = self._context_attention_kernel(_q, _cache_kv, new_infer_state, layer_weight)
                assert tmp_o.shape == _o.shape
                _o.copy_(tmp_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
        else:
            o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)

        return o
