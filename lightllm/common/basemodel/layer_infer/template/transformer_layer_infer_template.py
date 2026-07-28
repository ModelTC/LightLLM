import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from lightllm.distributed import all_reduce, all_reduce_fused_add_rmsnorm
from typing import Tuple
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


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
        # Subclasses that use a plain RMSNorm after the post-attention residual add
        # (gamma tensor at ``layer_weight.ffn_norm_weight_.weight``) can set this True
        # to fuse all_reduce + residual add + RMSNorm via FlashInfer. See
        # _should_fuse_ar_add_norm / _reduce_add_ffn_norm.
        self._enable_fused_ar_add_norm = False
        return

    def _should_fuse_ar_add_norm(self, infer_state: InferStateInfo) -> bool:
        # Only the plain-TP all-reduce path is fusible: skip under tpsp mix mode
        # (reduce-scatter) and when the FlashInfer all-reduce backend is absent.
        if not self._enable_fused_ar_add_norm or self.tp_world_size_ <= 1:
            return False
        args = get_env_start_args()
        if args.enable_tpsp_mix_mode or args.disable_fused_allreduce_norm:
            return False
        return getattr(infer_state.dist_group, "flashinfer_reduce", None) is not None

    def _reduce_add_ffn_norm(self, o, input_embdings, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        # o is the pre-reduce partial o_proj output (reduce was deferred in _get_o).
        # Fuse all_reduce + (input_embdings += o) + ffn_norm, else do them separately.
        # ponytail: only the post-attention reduce+add+norm is fused. The post-ffn
        # reduce+add is followed by the *next* layer's att_norm, which lives in a
        # different token_forward call, so it is left unfused. Fusing it too would
        # need threading a separate residual across the layer loop (vLLM-style) --
        # a much larger, riskier refactor for the second half of the win.
        o = o.view(-1, self.embed_dim_)
        flashinfer_reduce = getattr(infer_state.dist_group, "flashinfer_reduce", None)
        if flashinfer_reduce is None or not flashinfer_reduce.should_use(o):
            all_reduce(o, group=infer_state.dist_group)
            input_embdings.add_(o)
            return self._ffn_norm(input_embdings, infer_state, layer_weight)

        norm_out = self.alloc_tensor(o.shape, o.dtype)
        fused = all_reduce_fused_add_rmsnorm(
            o, input_embdings, layer_weight.ffn_norm_weight_.weight, self.eps_, norm_out, group=infer_state.dist_group
        )
        if fused:
            return norm_out
        all_reduce(o, group=infer_state.dist_group)
        input_embdings.add_(o)
        return self._ffn_norm(input_embdings, infer_state, layer_weight)

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn_norm(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_qkv(self, input, infer_state: InferStateInfo, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")

    def _post_cache_kv(self, cache_kv, infer_state: InferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        mem_manager.operator.copy_kv_to_mem_manager(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=cache_kv,
        )
        return

    def _context_attention_kernel(self, q, kv, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _token_attention_kernel(self, q, infer_state: InferStateInfo, layer_weight, out=None) -> torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight, defer_reduction=False) -> torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight) -> torch.Tensor:
        raise Exception("need to impl")

    def context_attention_forward(
        self, input_embdings, infer_state: InferStateInfo, layer_weight, defer_reduction=False
    ):
        q, cache_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )
        q = None
        if defer_reduction:
            o = self._get_o(o, infer_state, layer_weight, defer_reduction=True)
        else:
            o = self._get_o(o, infer_state, layer_weight)

        return o

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        use_fused_reduce = self._should_fuse_ar_add_norm(infer_state)
        if use_fused_reduce:
            o = self.context_attention_forward(input1, infer_state, layer_weight, defer_reduction=True)
            input1 = self._reduce_add_ffn_norm(o, input_embdings, infer_state, layer_weight)
        else:
            o = self.context_attention_forward(input1, infer_state, layer_weight)
            input_embdings.add_(o.view(-1, self.embed_dim_))
            input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        o = None

        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None

        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_attention_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight, defer_reduction=False):
        q, cache_kv = self._get_qkv(input_embdings, infer_state, layer_weight)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        if defer_reduction:
            o = self._get_o(o, infer_state, layer_weight, defer_reduction=True)
        else:
            o = self._get_o(o, infer_state, layer_weight)

        return o

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        use_fused_reduce = self._should_fuse_ar_add_norm(infer_state)
        if use_fused_reduce:
            o = self.token_attention_forward(input1, infer_state, layer_weight, defer_reduction=True)
            input1 = self._reduce_add_ffn_norm(o, input_embdings, infer_state, layer_weight)
        else:
            o = self.token_attention_forward(input1, infer_state, layer_weight)
            input_embdings.add_(o.view(-1, self.embed_dim_))
            input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        o = None

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
