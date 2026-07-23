import torch
import torch.distributed as dist
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from lightllm.utils.vllm_utils import vllm_ops
from .hyper_connection import hc_pre, hc_fused_post_pre, hc_post
from .compressor import fused_compress as fused_compress_op
from .compressor import apply_ape
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from ..infer_struct import DeepseekV4InferStateInfo
import deep_gemm
from lightllm.models.deepseek_v4.triton_kernel.topk_transform import topk_transform_512


_C4_PREFILL_LOGITS_BUDGET_BYTES = 512 * 1024 * 1024


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
        self.swiglu_limit = float(network_config["swiglu_limit"])
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        self.tp_q_head_num_ = self.num_heads // self.tp_world_size_
        self.flashmla_q_head_num_ = self.tp_q_head_num_
        self.tp_groups = self.o_groups // self.tp_world_size_
        self.enable_ep_moe = get_env_start_args().enable_ep_moe
        self.compressor = CompressorInfer(
            layer_idx=self.layer_num_, network_config=self.network_config_, tp_world_size=self.tp_world_size_
        )
        self.index_infer = DeepseekV4IndexInfer(
            layer_idx=self.layer_num_, network_config=self.network_config_, tp_world_size=self.tp_world_size_
        )
        self.dsv4_prefill_aux_stream = None

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
        out = self._hc_ffn_out(x, residual, post_mix, res_mix)
        return out

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
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope_cuda import fused_q_norm_rope

        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        T = input.shape[0]

        qkv = layer_weight.wq_a_wkv_.mm(input)
        qa = layer_weight.q_norm_(qkv[:, : -self.head_dim_], eps=self.eps_)
        q_in = layer_weight.wq_b_.mm(qa).view(T, self.tp_q_head_num_, self.head_dim_)

        if infer_state.is_prefill:
            q = infer_state.dsv4_workspace.flashmla_prefill_q[:T]
        else:
            q = self.alloc_tensor((T, self.flashmla_q_head_num_, self.head_dim_), dtype=q_in.dtype, device=q_in.device)
            q[:, self.tp_q_head_num_ :, :].zero_()
        fused_q_norm_rope(q_in, q[:, : self.tp_q_head_num_, :], self.eps_, self.freqs_cis, infer_state.position_ids)
        # kv: rmsnorm + rope + fp8 pack + scatter 进 swa 池,一个 DSV4 CUDA kernel 完成，
        infer_state.mem_manager.pack_mla_kv_to_cache_fused_norm_rope(
            layer_index=self.layer_num_,
            mem_index=infer_state.mem_index,
            kv=qkv[:, -self.head_dim_ :],
            kv_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            positions=infer_state.position_ids,
        )
        return q, qa, input

    def _get_o(self, o, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        # o: [T, tp_q_head_num_, head_dim_] after inverse rope -> grouped low-rank O -> [T, embed_dim_]
        position_cos, position_sin = self._select_rope(infer_state)
        rotary_emb_fwd(o[..., -self.qk_rope_head_dim :], None, position_cos, position_sin, inverse=True)
        T = o.shape[0]
        if layer_weight.o_proj_fp8:
            # one group per rank -> a single fp8 GEMM (deepgemm .mm quantizes o to fp8 internally)
            o = layer_weight.wo_a_.mm(o.reshape(T, -1))  # [T, o_lora]
        else:
            o = o.reshape(T, self.tp_groups, -1).transpose(0, 1).contiguous()  # [groups, T, per_group_in]
            o = layer_weight.wo_a_.bmm(o).transpose(0, 1).reshape(T, -1)  # [T, groups*o_lora]
        o = layer_weight.wo_b_.mm(o)
        return self._tpsp_reduce(input=o, infer_state=infer_state)

    # ------------------------------------------------------------------ attention (prefill)
    def context_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        # _get_qkv writes the chunk's packed latent into the swa pool (fused kernel) before
        # attention reads it back via full_to_swa indices (this custom forward bypasses the
        # tpl _post_cache_kv path).
        q, q_lora, full_x = self._get_qkv(x, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(q, q_lora, full_x, infer_state, layer_weight)
        return self._get_o(o, infer_state, layer_weight)

    def _context_attention_wrapper_run(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        if torch.cuda.is_current_stream_capturing():
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
                self._context_attention_kernel(_q, _q_lora, _x, new_infer_state, layer_weight, out=_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
            return o

        return self._context_attention_kernel(q, q_lora, x, infer_state, layer_weight)

    def _compress_and_index(self, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight):
        cos_table, sin_table = self.cos_compress_table, self.sin_compress_table
        aux_stream = self.dsv4_prefill_aux_stream
        if self.compress_ratio == 4 and aux_stream is not None and not torch.cuda.is_current_stream_capturing():
            main_stream = torch.cuda.current_stream()
            aux_stream.wait_stream(main_stream)  # aux waits for x / q_lora produced on main
            with torch.cuda.stream(aux_stream):
                # x / q_lora are main-allocated and read here -> record so the allocator won't reuse them.
                x.record_stream(aux_stream)
                q_lora.record_stream(aux_stream)
                self.index_infer.write_indexer_k(
                    x, infer_state, layer_weight, cos_table, sin_table, use_custom_tensor_manager=False
                )
                meta = self.index_infer.build_metadata(
                    x, q_lora, infer_state, layer_weight, use_custom_tensor_manager=False
                )
            self.compressor.compress(x, infer_state, layer_weight, cos_table, sin_table)
            main_stream.wait_stream(aux_stream)  # join before prefill_att reads the indices / latent KV
            # extra_indices / extra_lengths were allocated on aux -> record on main so they survive until consumed.
            for _t in (meta.get("extra_indices"), meta.get("extra_lengths")):
                if _t is not None:
                    _t.record_stream(main_stream)
            return meta

        self.compressor.compress(x, infer_state, layer_weight, cos_table, sin_table)
        # write c4 Lightning-Indexer keys BEFORE build_metadata so the scorer reads fresh+accumulated entries.
        self.index_infer.write_indexer_k(x, infer_state, layer_weight, cos_table, sin_table)
        return self.index_infer.build_metadata(x, q_lora, infer_state, layer_weight)

    def _context_attention_kernel(
        self,
        q,
        q_lora,
        x,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
        out=None,
    ):
        meta = self._compress_and_index(q_lora, x, infer_state, layer_weight)
        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "attn_sink": layer_weight.attn_sink_.weight,
                **meta,
            },
        )
        attn_out = infer_state.prefill_att_state.prefill_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
            alloc_func=self.alloc_tensor,
            out=out,
        )
        pad_q_len = getattr(infer_state, "_dsv4_prefill_pad_q_len", 0)
        if pad_q_len:
            # pad 行读 HOLD 槽位(参见 infer_struct._dsv4_prefill_pad_q_len),清零以保持确定性
            attn_out[-pad_q_len:] = 0
        return attn_out

    # ------------------------------------------------------------------ attention (decode)
    def token_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        q, q_lora, full_x = self._get_qkv(x, infer_state, layer_weight)
        o = self._token_attention_kernel(q, q_lora, full_x, infer_state, layer_weight)
        return self._get_o(o, infer_state, layer_weight)

    def _token_attention_kernel(
        self, q, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        self.compressor.compress(x, infer_state, layer_weight, self.cos_compress_table, self.sin_compress_table)
        self.index_infer.write_indexer_k(x, infer_state, layer_weight, self.cos_compress_table, self.sin_compress_table)
        meta = self.index_infer.build_metadata(x, q_lora, infer_state, layer_weight)
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "layer_index": self.layer_num_,
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "attn_sink": layer_weight.attn_sink_.weight,
                **meta,
            },
        )
        return infer_state.decode_att_state.decode_att(
            q=q,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )

    # ------------------------------------------------------------------ moe
    def _routed_experts(
        self,
        x,
        weights,
        indices,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        return layer_weight.experts_.experts_with_topk(
            input_tensor=x,
            topk_weights=weights,
            topk_ids=indices,
            is_prefill=infer_state.is_prefill,
            clamp_limit=float(self.swiglu_limit),
            alloc_tensor_func=self.alloc_tensor,
        )

    def _ffn_tp(self, input, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        input = input.view(-1, self.embed_dim_)
        gate_up = layer_weight.gate_up_proj.mm(input)
        shared = self.alloc_tensor((input.size(0), gate_up.size(1) // 2), input.dtype)
        silu_and_mul_fwd(gate_up, shared, limit=self.swiglu_limit)
        input = None
        gate_up = None
        out = layer_weight.down_proj.mm(shared)
        shared = None
        return out

    def _ffn(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        x = x.view(-1, self.embed_dim_)
        if not self.enable_ep_moe:
            x = self._tpsp_allgather(input=x, infer_state=infer_state)

        logits = layer_weight.gate_weight_.mm(x, out_dtype=torch.float32)
        weights, indices = self._select_experts(logits, infer_state, layer_weight)
        # shared expert 必须先于 routed 计算: fp8 路径 (FuseMoeTriton) 的 fused_experts
        # 是 inplace 的，_routed_experts 返回后 x 已被覆盖为 routed 输出。
        # DS4 shared experts also use the config swiglu_limit clamp, matching SGLang's
        # DeepseekV2MLP(..., swiglu_limit=config.swiglu_limit) path.
        shared = self._ffn_tp(input=x, infer_state=infer_state, layer_weight=layer_weight)
        routed = self._routed_experts(x, weights, indices, infer_state, layer_weight)
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
        M = logits.shape[0]
        bias = None
        input_tokens = None
        hash_indices_table = None
        indices_dtype = torch.int64
        if self.is_hash:
            hash_indices_table = layer_weight.gate_tid2eid_.weight
            indices_dtype = hash_indices_table.dtype
            input_tokens = infer_state.input_ids.to(dtype=indices_dtype)
        else:
            bias = layer_weight.gate_bias_.weight

        weights = self.alloc_tensor((M, self.num_experts_per_tok), dtype=torch.float32, device=logits.device)
        indices = self.alloc_tensor((M, self.num_experts_per_tok), dtype=indices_dtype, device=logits.device)
        token_expert_indices = self.alloc_tensor((M, self.num_experts_per_tok), dtype=torch.int32, device=logits.device)
        vllm_ops.topk_hash_softplus_sqrt(
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
        return weights, indices


class CompressorInfer:
    """Window-softmax compressor. is_in_indexer=False compresses the c4/c128 latent KV into the
    paged fp8 slab (attention extra_k); is_in_indexer=True reuses the SAME machinery (mirroring
    sglang's Compressor(is_in_indexer=...)) with the indexer weights/dims/state pool to produce the
    per-c4-entry Lightning-Indexer keys, emitted as dense bf16 (OUTPUT_BF16) then fp8-packed into
    c4_indexer_pool by the caller. Indexer mode is c4-only."""

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int, is_in_indexer: bool = False):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.tp_world_size_ = tp_world_size
        self.is_in_indexer = is_in_indexer
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.head_dim = network_config["head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.eps = network_config["rms_norm_eps"]

    def compress(
        self,
        x: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        use_custom_tensor_manager: bool = True,
    ):
        if self.compress_ratio == 0:
            return None
        if self.is_in_indexer:
            kv_score = layer_weight.idx_cmp_wkv_gate_.mm(
                x, use_custom_tensor_mananger=use_custom_tensor_manager, out_dtype=torch.float32
            )
            norm_weight = layer_weight.idx_cmp_norm_.weight
            ape = layer_weight.idx_cmp_ape_.weight
            head_dim = self.index_head_dim
        else:
            kv_score = layer_weight.compressor_wkv_gate_.mm(
                x, use_custom_tensor_mananger=use_custom_tensor_manager, out_dtype=torch.float32
            )
            norm_weight = layer_weight.compressor_norm_.weight
            ape = layer_weight.compressor_ape_.weight
            head_dim = self.head_dim
        apply_ape(
            kv_score=kv_score,
            position_ids=infer_state.position_ids,
            ape=ape,
            compress_ratio=self.compress_ratio,
        )
        return fused_compress_op(
            kv_score=kv_score,
            infer_state=infer_state,
            layer_idx=self.layer_idx_,
            norm_weight=norm_weight,
            eps=self.eps,
            head_dim=head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            compress_ratio=self.compress_ratio,
            cos_table=cos_table,
            sin_table=sin_table,
            is_in_indexer=self.is_in_indexer,
        )


class DeepseekV4IndexInfer:
    """Model-side builder for the FlashMLA sparse-index metadata. Mirrors deepseek3_2's NsaInfer
    boundary (the model owns ALL index construction; the attention backend only forwards final
    tensors to flash_mla.flash_mla_with_kvcache) AND its c4 implementation: hadamard'd fp8 q/K, a
    ragged gather of the compressed c4 keys, deep_gemm.fp8_mqa_logits, then topk -- adapted for the
    replicated indexer (no gather-q/all_reduce), the c4-compressed entry space, and topk-512 (no
    inheritance only because of those data-shape differences). swa metadata is precomputed in
    init_some_extra_state; this class owns the c4 entry gather (build_compress_index) AND the c4
    Lightning-Indexer scoring (gather + deep_gemm.fp8_mqa_logits + topk). Holds only static per-layer
    config; all per-request data flows in via args. Invoke from _context/_token_attention_kernel
    (after compressor.compress, before *_att) so the c4 scorer/topk keep the same cuda-graph
    capture position they had when this lived in the backend. The indexer is replicated (no TP collective)."""

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int):
        self.layer_idx_ = layer_idx
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.index_topk = network_config["index_topk"]
        self.index_head_dim = network_config["index_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.index_n_heads = network_config["index_n_heads"]
        self.tp_world_size_ = tp_world_size
        self.indexer_score_scale = self.index_head_dim ** -0.5
        self.indexer_weight_scale = self.indexer_score_scale * self.index_n_heads ** -0.5
        # c4 layers own a second compressor (is_in_indexer) that writes the Lightning-Indexer key
        # pool every step; _c4_indices gathers it back + scores via deep_gemm.fp8_mqa_logits.
        self.indexer_compressor = (
            CompressorInfer(layer_idx, network_config, tp_world_size, is_in_indexer=True)
            if self.compress_ratio == 4
            else None
        )

    def write_indexer_k(
        self,
        x,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight,
        cos_table,
        sin_table,
        use_custom_tensor_manager=True,
    ):
        if self.compress_ratio != 4:
            return
        # Only group-end rows in this dense bf16 scratch are valid indexer keys.
        scratch = self.indexer_compressor.compress(
            x,
            infer_state,
            layer_weight,
            cos_table,
            sin_table,
            use_custom_tensor_manager=use_custom_tensor_manager,
        )
        # Rotate K (post norm+rope) by the SAME 1/sqrt(d) Hadamard the q kernel applies, so
        # (Hq)·(Hk)=q·k (H orthogonal) and the fp8 quant of K stays accurate.
        from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import hadamard_transform

        scratch = hadamard_transform(scratch, scale=self.index_head_dim ** -0.5)
        mem_manager = infer_state.mem_manager
        mem_manager.pack_indexer_k_to_cache(
            self.layer_idx_,
            infer_state.mem_index.reshape(-1),
            infer_state.position_ids,
            scratch,
        )

    def build_metadata(
        self, x, q_lora, infer_state: DeepseekV4InferStateInfo, layer_weight, use_custom_tensor_manager=True
    ):
        """Return the final flash_mla index tensors for this layer's compress variant. swa indices and
        the per-token req_idx are layer-independent and precomputed once in init_some_extra_state
        (read here); only the c4 scorer is per-layer. The backend pairs these with the
        (data-independent, layer-keyed) fp8 cache-byte views it owns."""
        swa_indices = infer_state.dsv4_swa_indices.unsqueeze(1)
        swa_lengths = infer_state.dsv4_swa_lengths
        positions = infer_state.position_ids
        extra_indices = extra_lengths = None
        if self.compress_ratio == 4:
            idx_q_fp8, weights = self._indexer_q_weight(
                x, q_lora, infer_state, layer_weight, use_custom_tensor_manager=use_custom_tensor_manager
            )
            extra_indices, extra_lengths = self._c4_indices(infer_state, idx_q_fp8, weights, positions)
        elif self.compress_ratio == 128:
            extra_indices = infer_state.dsv4_c128_indices.unsqueeze(1)
            extra_lengths = infer_state.dsv4_c128_lengths
        return {
            "swa_indices": swa_indices,
            "swa_lengths": swa_lengths,
            "extra_indices": extra_indices,
            "extra_lengths": extra_lengths,
        }

    def _indexer_q_weight(
        self, x, q_lora, infer_state: DeepseekV4InferStateInfo, layer_weight, use_custom_tensor_manager=True
    ):
        # Fused: wq_b mm -> rope(last rope dims) -> 1/sqrt(d) Hadamard -> per-token fp8 quant, with the
        # per-token q scale + indexer_weight_scale folded into weights, all in ONE kernel (was 4 kernels:
        # rotary_emb_fwd + hadamard_transform + act_quant + weights mul). freqs_cis is the compress rope
        # table (same one the main compress-layer Q path uses); positions indexed inside the kernel.
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope_cuda import (
            fused_q_indexer_rope_hadamard_quant,
        )

        token_num = q_lora.shape[0]
        if x.shape[0] != token_num:
            raise RuntimeError(
                f"DeepSeek-V4 indexer expects full-token hidden states, got x={x.shape[0]} q_lora={token_num}"
            )
        idx_q = layer_weight.idx_wq_b_.mm(q_lora, use_custom_tensor_mananger=use_custom_tensor_manager).view(
            token_num, self.index_n_heads, self.index_head_dim
        )
        raw_w = layer_weight.idx_weights_proj_.mm(x, use_custom_tensor_mananger=use_custom_tensor_manager).view(
            token_num, self.index_n_heads
        )  # [T, H] raw
        idx_q_fp8, weights = fused_q_indexer_rope_hadamard_quant(
            idx_q,
            raw_w,
            self.indexer_weight_scale,
            self.freqs_cis,
            infer_state.position_ids,
        )  # fp8 [T,H,d]; weights [T,H,1] with q-scale + weight_scale folded
        return idx_q_fp8, weights.squeeze(-1)

    def _c4_indices(self, infer_state: DeepseekV4InferStateInfo, idx_q_fp8, weights, positions):
        """c4 scorer via the page-safe deep_gemm.fp8_paged_mqa_logits over the paged c4 indexer pool,
        then masked topk-512 -> c4 slots. Fixed shapes (c4_cap pinned per graph bucket) keep the decode
        cuda graph capturable."""
        mem_manager = infer_state.mem_manager
        workspace = infer_state.dsv4_workspace
        index_topk = self.index_topk
        max_entries = max(1, int(infer_state.max_kv_seq_len) // 4)
        c4_cap = ((max_entries + 63) // 64) * 64

        # entry space fits the budget -> every causal entry is selected; no scoring needed. The
        # captured decode graph (graph_max_len -> max_entries > topk) always takes the scorer branch
        # below, so this only shortcuts tiny eager contexts.
        if max_entries <= index_topk:
            from ..triton_kernel.build_compress_index_dsv4 import build_compress_index

            slots, lengths = workspace.c4(infer_state.microbatch_index, positions.shape[0], c4_cap)
            slots, lengths = build_compress_index(
                infer_state.dsv4_sparse_req_idx,
                positions,
                infer_state.req_manager.req_to_token_indexs,
                mem_manager.full_to_c4_indexs,
                4,
                slots,
                lengths,
            )
            return slots.unsqueeze(1), lengths

        c4_len = torch.div(infer_state.b_seq_len, 4, rounding_mode="floor").to(torch.int32)  # entries/req

        device = positions.device
        page_size = mem_manager.c4_indexer_pool.page_size

        # The page table / row_page_table / valid_len / ctx_lens / paged-logits metadata / topk_lengths
        # are LAYER-INDEPENDENT (depend on request layout + c4_cap, not on weights/layer). Build them on
        # the first c4 layer of the forward and reuse on the other ~20 c4 layers (was rebuilt per layer:
        # build_c4_indexer_page_table + a [T,npages] gather + clamp/reshape + get_paged_mqa_logits_metadata
        # each, i.e. ~20x redundant index/copy/clamp launches). Lazy (not init_some_extra_state) so it is
        # computed inside the decode cuda graph with the capture-forced shapes -> no graph-cap mismatch.
        cached = getattr(infer_state, "_c4_paged_meta", None)
        if cached is None:
            from ..triton_kernel.gather_c4_indexer_k_dsv4 import build_c4_indexer_page_table

            b_req_idx = infer_state.b_req_idx
            batch = b_req_idx.shape[0]
            page_table = build_c4_indexer_page_table(
                mem_manager,
                b_req_idx,
                c4_len,
                c4_cap,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.req_manager.HOLD_REQUEST_ID,
            )

            if infer_state.is_prefill:
                token_batch_pos = torch.repeat_interleave(
                    torch.arange(batch, device=device, dtype=torch.int32),
                    infer_state.b_q_seq_len,
                    output_size=positions.numel(),
                )
                row_page_table = page_table[token_batch_pos]
            else:
                row_page_table = page_table

            valid_len = ((positions + 1) // 4).to(torch.int32)
            ctx_lens = torch.clamp(valid_len, min=1).reshape(-1, 1)
            metadata = deep_gemm.get_paged_mqa_logits_metadata(
                ctx_lens,
                page_size,
                deep_gemm.get_num_sms(),
            )
            topk_lengths = torch.clamp(torch.minimum(valid_len, torch.full_like(valid_len, index_topk)), min=1)
            cached = (row_page_table, valid_len, ctx_lens, metadata, topk_lengths)
            infer_state._c4_paged_meta = cached

        row_page_table, valid_len, ctx_lens, metadata, topk_lengths = cached
        kv_cache = mem_manager.c4_indexer_pool.get_layer_buffer(mem_manager.layer_to_c4_idx[self.layer_idx_]).view(
            mem_manager.c4_indexer_pool.num_pages,
            page_size,
            1,
            self.index_head_dim + 4,
        )
        top_slots, _ = workspace.c4(infer_state.microbatch_index, idx_q_fp8.shape[0], index_topk)
        if infer_state.is_prefill:
            rows_per_chunk = max(1, _C4_PREFILL_LOGITS_BUDGET_BYTES // (c4_cap * 4))
            if idx_q_fp8.shape[0] > rows_per_chunk:
                for start in range(0, idx_q_fp8.shape[0], rows_per_chunk):
                    end = min(start + rows_per_chunk, idx_q_fp8.shape[0])
                    chunk_ctx_lens = ctx_lens[start:end]
                    self._c4_score_topk(
                        idx_q_fp8[start:end],
                        kv_cache,
                        weights[start:end],
                        chunk_ctx_lens,
                        row_page_table[start:end],
                        deep_gemm.get_paged_mqa_logits_metadata(
                            chunk_ctx_lens,
                            page_size,
                            deep_gemm.get_num_sms(),
                        ),
                        c4_cap,
                        valid_len[start:end],
                        top_slots[start:end],
                        page_size,
                    )
                return top_slots.unsqueeze(1), topk_lengths

        self._c4_score_topk(
            idx_q_fp8,
            kv_cache,
            weights,
            ctx_lens,
            row_page_table,
            metadata,
            c4_cap,
            valid_len,
            top_slots,
            page_size,
        )
        return top_slots.unsqueeze(1), topk_lengths

    @staticmethod
    def _c4_score_topk(
        idx_q_fp8,
        kv_cache,
        weights,
        ctx_lens,
        row_page_table,
        metadata,
        c4_cap,
        valid_len,
        top_slots,
        page_size,
    ):
        logits = deep_gemm.fp8_paged_mqa_logits(
            idx_q_fp8.unsqueeze(1),
            kv_cache,
            weights,
            ctx_lens,
            row_page_table,
            metadata,
            c4_cap,
            False,
        )
        topk_transform_512(
            logits,
            valid_len,
            row_page_table,
            top_slots,
            page_size,
        )
