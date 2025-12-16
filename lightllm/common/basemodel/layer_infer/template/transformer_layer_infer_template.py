import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.distributed import all_reduce
from typing import Tuple
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class TransformerLayerInferTpl(TransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode):
        super().__init__(layer_num, network_config, mode)
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

    def _tpsp_get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _post_cache_kv(self, cache_kv, infer_state: InferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        self._copy_kv_to_mem_cache(cache_kv, infer_state.mem_index, mem_manager)
        return

    def _copy_kv_to_mem_cache(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        return

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _tpsp_get_o(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _tpsp_ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)

        # prefill 的 cuda graph 过程， 排除掉attention部分
        if torch.cuda.is_current_stream_capturing():
            infer_state.cpu_prefill_atomic_event += 1
            infer_state.prefill_atomic_event_incr()
            wait_event = infer_state.cpu_prefill_atomic_event
            o = torch.empty_like(q)
            _q, _cache_kv, _o = (
                tensor_to_no_ref_tensor(q.contiguous()),
                tensor_to_no_ref_tensor(cache_kv.contiguous()),
                tensor_to_no_ref_tensor(o),
            )

            def att_func(new_infer_state: InferStateInfo):
                infer_state.prefill_atomic_event_wait(wait_event=wait_event)
                _o.copy_(self._context_attention_kernel(_q, _cache_kv, new_infer_state, layer_weight))
                infer_state.prefill_atomic_event_incr()
                return

            infer_state.cpu_prefill_atomic_event += 1
            infer_state.prefill_atomic_event_wait(wait_event=infer_state.cpu_prefill_atomic_event)

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func)
        else:
            o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)

        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def tpsp_context_forward(self, input_embdings: torch.Tensor, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._tpsp_get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)

        # prefill 的 cuda graph 过程， 排除掉attention部分
        if torch.cuda.is_current_stream_capturing():
            infer_state.cpu_prefill_atomic_event += 1
            infer_state.prefill_atomic_event_incr()
            wait_event = infer_state.cpu_prefill_atomic_event
            o = torch.empty_like(q)
            _q, _cache_kv, _o = (
                tensor_to_no_ref_tensor(q.contiguous()),
                tensor_to_no_ref_tensor(cache_kv.contiguous()),
                tensor_to_no_ref_tensor(o),
            )

            def att_func(new_infer_state: InferStateInfo):
                infer_state.prefill_atomic_event_wait(wait_event=wait_event)
                _o.copy_(self._context_attention_kernel(_q, _cache_kv, new_infer_state, layer_weight))
                infer_state.prefill_atomic_event_incr()
                return

            infer_state.cpu_prefill_atomic_event += 1
            infer_state.prefill_atomic_event_wait(wait_event=infer_state.cpu_prefill_atomic_event)

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func)
        else:
            o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)

        q = None
        o = self._tpsp_get_o(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._tpsp_ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def tpsp_token_forward(self, input_embdings: torch.Tensor, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._tpsp_get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._tpsp_get_o(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._tpsp_ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
