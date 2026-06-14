import torch
import torch.nn.functional as F
import torch.distributed as dist
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from .hyper_connection import hc_pre, hc_fused_post_pre, hc_post
from .compressor import fused_compress as fused_compress_op
from .compressor import prepare_partial_states
from .compressor import prepare_compress_states
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from ..infer_struct import DeepseekV4InferStateInfo


class DeepseekV4TransformerLayerInfer(Deepseek3_2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        TransformerLayerInferTpl.__init__(self, layer_num, network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.embed_dim_ = network_config["hidden_size"]
        self.num_heads = network_config["num_attention_heads"]
        self.head_dim_ = network_config["head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.qk_nope_head_dim = self.head_dim_ - self.qk_rope_head_dim
        self.v_head_dim = self.head_dim_
        self.index_n_heads = network_config["index_n_heads"]
        self.index_head_dim = network_config["index_head_dim"]
        self.index_topk = network_config["index_topk"]
        self.o_groups = network_config["o_groups"]
        self.hc_mult = network_config["hc_mult"]
        self.sinkhorn_iters = network_config["hc_sinkhorn_iters"]
        self.hc_eps = network_config["hc_eps"]
        self.compress_ratio = network_config["compress_ratios"][layer_num]
        self.is_hash = layer_num < network_config["num_hash_layers"]
        self.is_last_layer = layer_num == network_config["n_layer"] - 1
        # complex64 rope table for this layer's variant (sliding / compressed); set by
        # DeepseekV4TpPartModel._init_to_get_rotary once the tables are built. The full compress
        # cos/sin tables (compressor entry rope uses entry positions, not token positions) are
        # wired there too.
        self.freqs_cis = None
        self.cos_compress_table = None
        self.sin_compress_table = None
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.routed_scaling_factor = network_config["routed_scaling_factor"]
        self.swiglu_limit = network_config["swiglu_limit"]
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.tp_q_head_num_ = self.num_heads // self.tp_world_size_
        self.tp_index_n_heads = self.index_n_heads // self.tp_world_size_
        self.tp_groups = self.o_groups // self.tp_world_size_
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self.indexer_score_scale = self.index_head_dim ** -0.5
        self.indexer_weight_scale = self.indexer_score_scale * self.index_n_heads ** -0.5
        self.compressor = CompressorInfer(
            layer_idx=self.layer_num_, network_config=self.network_config_, tp_world_size=self.tp_world_size_
        )

    # ------------------------------------------------------------------ forward (HC-threaded)
    def _hc_attn_in(self, input_embdings, layer_weight: DeepseekV4TransformerLayerWeight):
        """Layer input -> attention input (attn_norm fused). First layer gets the raw streams
        and runs a standalone hc_pre; later layers get (x, residual, post_mix, res_mix) and fuse
        the previous layer's ffn hc_post with this layer's attn hc_pre."""
        if torch.is_tensor(input_embdings):
            residual = input_embdings.view(-1, self.hc_mult, self.embed_dim_)
            return hc_pre(
                residual,
                layer_weight.hc_attn_fn_.weight,
                layer_weight.hc_attn_scale_.weight,
                layer_weight.hc_attn_base_.weight,
                self.eps_,
                self.hc_eps,
                self.sinkhorn_iters,
                layer_weight.attn_norm_.weight,
                self.eps_,
            )
        x, residual, post_mix, res_mix = input_embdings
        return hc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            layer_weight.hc_attn_fn_.weight,
            layer_weight.hc_attn_scale_.weight,
            layer_weight.hc_attn_base_.weight,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            layer_weight.attn_norm_.weight,
            self.eps_,
        )

    def _hc_ffn_in(self, x, residual, post_mix, res_mix, layer_weight: DeepseekV4TransformerLayerWeight):
        """Attention output -> ffn input (ffn_norm fused): fused attn hc_post + ffn hc_pre."""
        return hc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            layer_weight.hc_ffn_fn_.weight,
            layer_weight.hc_ffn_scale_.weight,
            layer_weight.hc_ffn_base_.weight,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            layer_weight.ffn_norm_.weight,
            self.eps_,
        )

    def _hc_ffn_out(self, x, residual, post_mix, res_mix):
        """Mid layers leave the ffn hc_post pending for the next layer's fused post+pre; the last
        layer completes it and hands the flat streams [T, hc_mult*hidden] back to the model loop."""
        if not self.is_last_layer:
            return x, residual, post_mix, res_mix
        streams = hc_post(x, residual, post_mix, res_mix)
        return streams.reshape(streams.shape[0], -1)

    def context_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, post_mix, res_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.context_attention_forward(x, infer_state, layer_weight)
        x, residual, post_mix, res_mix = self._hc_ffn_in(x, residual, post_mix, res_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, post_mix, res_mix)

    def token_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, post_mix, res_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.token_attention_forward(x, infer_state, layer_weight)
        x, residual, post_mix, res_mix = self._hc_ffn_in(x, residual, post_mix, res_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, post_mix, res_mix)

    # ------------------------------------------------------------------ shared projections / cache
    def _select_rope(self, infer_state: DeepseekV4InferStateInfo):
        if self.compress_ratio:
            return infer_state.position_cos_compress, infer_state.position_sin_compress
        return infer_state.position_cos_sliding, infer_state.position_sin_sliding

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        from sglang.jit_kernel.dsv4 import fused_q_norm_rope

        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        T = input.shape[0]
        qa = layer_weight.q_norm_(layer_weight.wq_a_.mm(input), eps=self.eps_)
        q_in = layer_weight.wq_b_.mm(qa).view(T, self.tp_q_head_num_, self.head_dim_)
        # per-(token, head) weightless self-RMSNorm + interleaved rope on the last rope_dim dims,
        # fused in one sglang dsv4 jit kernel (fp32 norm/rotation, bf16 in between -- same as eager).
        q = self.alloc_tensor(q_in.shape, dtype=q_in.dtype, device=q_in.device)
        fused_q_norm_rope(q_in, q, self.eps_, self.freqs_cis, infer_state.position_ids)
        # kv: rmsnorm + rope + fp8 pack + scatter 进 swa 池,一个 sglang jit kernel 完成
        # (同 sglang _compute_kv_to_cache),替代 eager norm/rope/cat + _post_cache_kv。
        # bf16 kv 中间量没有其他消费者: flashmla 路径注意力读 cache,压缩器/indexer 取 x。
        infer_state.mem_manager.pack_mla_kv_to_cache_fused_norm_rope(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=layer_weight.wkv_.mm(input),
            kv_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            positions=infer_state.position_ids,
        )
        return q, qa

    def _get_o(self, o, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        # o: [T, tp_q_head_num_, head_dim_] after inverse rope -> grouped low-rank O -> [T, embed_dim_]
        position_cos, position_sin = self._select_rope(infer_state)
        rotary_emb_fwd(o[..., -self.qk_rope_head_dim :], None, position_cos, position_sin, inverse=True)
        T = o.shape[0]
        o = o.reshape(T, self.tp_groups, -1).transpose(0, 1).contiguous()  # [groups, T, per_group_in]
        o = layer_weight.wo_a_.bmm(o).transpose(0, 1).reshape(T, -1)  # [T, groups*o_lora]
        o = layer_weight.wo_b_.mm(o)
        return self._tpsp_reduce(input=o, infer_state=infer_state)

    # ------------------------------------------------------------------ compressor / indexer
    def _indexer_q_weight(
        self, x, q_lora, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        if self.compress_ratio != 4:
            return None, None
        cos_tok = infer_state.position_cos_compress
        sin_tok = infer_state.position_sin_compress
        idx_q = layer_weight.idx_wq_b_.mm(q_lora).view(x.shape[0], self.tp_index_n_heads, self.index_head_dim)
        rotary_emb_fwd(idx_q[..., -self.qk_rope_head_dim :], None, cos_tok, sin_tok)
        idx_weight = layer_weight.idx_weights_proj_.mm(x).float() * self.indexer_weight_scale
        return idx_q, idx_weight

    # ------------------------------------------------------------------ attention (prefill)
    def context_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        # _get_qkv writes the chunk's packed latent into the swa pool (fused kernel) before
        # attention reads it back via full_to_swa indices (this custom forward bypasses the
        # tpl _post_cache_kv path).
        q, q_lora = self._get_qkv(x, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(q, q_lora, x, infer_state, layer_weight)
        return self._get_o(o, infer_state, layer_weight)

    def _context_attention_wrapper_run(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        if torch.cuda.is_current_stream_capturing():
            q = q.contiguous()
            q_lora = q_lora.contiguous()
            x = x.contiguous()
            _q = tensor_to_no_ref_tensor(q)
            _q_lora = tensor_to_no_ref_tensor(q_lora)
            _x = tensor_to_no_ref_tensor(x)

            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()
            # Same graph-split output handoff as the template, but avoid its dry-run because
            # DSV4 attention mutates compressor/cache state before returning.
            o = self.alloc_tensor((q.shape[0], self.tp_q_head_num_, self.head_dim_), dtype=q.dtype, device=q.device)
            _o = tensor_to_no_ref_tensor(o)

            def att_func(new_infer_state: DeepseekV4InferStateInfo):
                tmp_o = self._context_attention_kernel(_q, _q_lora, _x, new_infer_state, layer_weight)
                assert tmp_o.shape == _o.shape
                _o.copy_(tmp_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
            return o

        return self._context_attention_kernel(q, q_lora, x, infer_state, layer_weight)

    def _context_attention_kernel(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        self.compressor.prepare_states(x, infer_state, layer_weight)
        self.compressor.fused_compress(infer_state, layer_weight, self.cos_compress_table, self.sin_compress_table)
        idx_q, idx_weight = self._indexer_q_weight(x, q_lora, infer_state, layer_weight)
        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "flashmla_kvcache": True,
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "q_lora": q_lora,
                "hidden_states": x,
                "attn_sink": layer_weight.attn_sink_.weight,
                "idx_q": idx_q,
                "idx_weight": idx_weight,
                "index_topk": self.index_topk,
                "indexer_score_scale": self.indexer_score_scale,
                "tp_world_size": self.tp_world_size_,
            },
        )
        out = infer_state.prefill_att_state.prefill_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )
        pad_q_len = getattr(infer_state, "_dsv4_prefill_pad_q_len", 0)
        if pad_q_len:
            # pad 行读 HOLD 槽位(参见 infer_struct._dsv4_prefill_pad_q_len),清零以保持确定性
            out[-pad_q_len:] = 0
        return out

    # ------------------------------------------------------------------ attention (decode)
    def token_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        q, q_lora = self._get_qkv(x, infer_state, layer_weight)
        o = self._token_attention_kernel(q, q_lora, x, infer_state, layer_weight)
        return self._get_o(o, infer_state, layer_weight)

    def _token_attention_kernel(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        self.compressor.prepare_states(x, infer_state, layer_weight)
        self.compressor.fused_compress(infer_state, layer_weight, self.cos_compress_table, self.sin_compress_table)
        idx_q, idx_weight = self._indexer_q_weight(x, q_lora, infer_state, layer_weight)
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "flashmla_kvcache": True,
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "q_lora": q_lora,
                "hidden_states": x,
                "attn_sink": layer_weight.attn_sink_.weight,
                "idx_q": idx_q,
                "idx_weight": idx_weight,
                "index_topk": self.index_topk,
                "indexer_score_scale": self.indexer_score_scale,
                "tp_world_size": self.tp_world_size_,
            },
        )
        return infer_state.decode_att_state.decode_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )

    # ------------------------------------------------------------------ moe
    def _routed_experts(self, x, weights, indices, layer_weight: DeepseekV4TransformerLayerWeight):
        return layer_weight.experts_.experts_with_preselected(
            input_tensor=x,
            topk_weights=weights,
            topk_ids=indices,
            clamp_limit=float(self.swiglu_limit),
        )

    def _ffn(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        x = x.view(-1, self.embed_dim_)
        if not self.enable_ep_moe:
            x = self._tpsp_allgather(input=x, infer_state=infer_state)

        gw = layer_weight.gate_weight_.mm_param.weight
        logits = F.linear(x.float(), gw.float()).contiguous()
        weights, indices = self._select_experts(logits, infer_state, layer_weight)
        # shared expert 必须先于 routed 计算: fp8 路径 (FuseMoeTriton) 的 fused_experts
        # 是 inplace 的，_routed_experts 返回后 x 已被覆盖为 routed 输出。
        g = layer_weight.shared_gate_.mm(x).float().clamp(max=self.swiglu_limit)
        u = layer_weight.shared_up_.mm(x).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        shared = layer_weight.shared_down_.mm((F.silu(g) * u).to(x.dtype))
        routed = self._routed_experts(x, weights, indices, layer_weight)
        if self.enable_ep_moe:
            if self.tp_world_size_ > 1:
                all_reduce(
                    shared,
                    op=dist.ReduceOp.SUM,
                    group=infer_state.dist_group,
                    async_op=False,
                )
            return routed + shared
        out = routed + shared
        return self._tpsp_reduce(input=out, infer_state=infer_state)

    def _select_experts(
        self, logits, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        return self._select_experts_vllm(logits, infer_state, layer_weight)

    def _select_experts_vllm(
        self, logits, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        from vllm import _custom_ops as ops

        M = logits.shape[0]
        bias = None
        input_tokens = None
        hash_indices_table = None
        indices_dtype = torch.int64
        if self.is_hash:
            hash_indices_table = layer_weight.gate_tid2eid_.weight
            if not hash_indices_table.is_contiguous():
                hash_indices_table = hash_indices_table.contiguous()
            indices_dtype = hash_indices_table.dtype
            input_tokens = infer_state.input_ids.to(dtype=indices_dtype).contiguous()
        else:
            bias = layer_weight.gate_bias_.weight

        weights = self.alloc_tensor((M, self.num_experts_per_tok), dtype=torch.float32, device=logits.device)
        indices = self.alloc_tensor((M, self.num_experts_per_tok), dtype=indices_dtype, device=logits.device)
        token_expert_indices = self.alloc_tensor((M, self.num_experts_per_tok), dtype=torch.int32, device=logits.device)
        ops.topk_hash_softplus_sqrt(
            weights,
            indices,
            token_expert_indices,
            logits,
            True,
            self.routed_scaling_factor,
            bias,
            input_tokens,
            hash_indices_table,
        )
        return weights, indices.long()


class CompressorInfer:
    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.tp_world_size_ = tp_world_size
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.head_dim = network_config["head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.eps = network_config["rms_norm_eps"]
        self._metadata = None

    def prepare_states(
        self,
        x: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        self._metadata = prepare_compress_states(
            infer_state=infer_state,
            layer_idx=self.layer_idx_,
            compress_ratio=self.compress_ratio,
        )
        if self._metadata is not None:
            self._metadata.kv_score = layer_weight.compressor_wkv_gate_.mm(x).float()
            prepare_partial_states(
                kv_score=self._metadata.kv_score,
                metadata=self._metadata,
                ape=layer_weight.compressor_ape_.weight,
                compress_ratio=self.compress_ratio,
            )
        return self._metadata

    def fused_compress(
        self,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ):
        if self.compress_ratio == 0:
            return None
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("DeepSeek-V4 compressor.prepare_states must run before fused_compress")
        return fused_compress_op(
            kv_score=metadata.kv_score,
            metadata=metadata,
            norm_weight=layer_weight.compressor_norm_.weight,
            ape=layer_weight.compressor_ape_.weight,
            eps=self.eps,
            head_dim=self.head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            compress_ratio=self.compress_ratio,
            cos_table=cos_table,
            sin_table=sin_table,
        )
