import torch
import triton
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.triton_kernel.mhc import hc_post, hc_post_pre_norm, hc_pre_norm
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import use_mega_moe
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_ep_impl import FuseMoeTritonEP
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor
from .compressor import CompressorInfer
from .indexer import DeepseekV4IndexInfer
from ..triton_kernel.rotary_emb import inverse_rotary_emb
from ..infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4.triton_kernel.topk_softplus_sqrt import topk_softplus_sqrt


class DeepseekV4TransformerLayerInfer(TransformerLayerInferTpl):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
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
        self.has_vision = network_config.get("vision_n_layers", 0) > 0
        self.vocab_size = network_config["vocab_size"]
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

    def _hc_attn_in(self, input_embdings, layer_weight: DeepseekV4TransformerLayerWeight):
        """Run attention pre + norm. Later layers also finish the previous FFN post.

        Residuals stay [T, S * H]. State order matches hc_post_pre_norm:
        (x, residual, residual_mix, post_mix).
        """
        if torch.is_tensor(input_embdings):
            residual = input_embdings
            x, residual_mix, post_mix = hc_pre_norm(
                x=residual,
                fn=layer_weight.hc_attn_fn_.weight,
                scale=layer_weight.hc_attn_scale_.weight,
                base=layer_weight.hc_attn_base_.weight,
                norm_weight=layer_weight.attn_norm_.weight,
                streams=self.hc_mult,
                rms_eps=self.eps_,
                norm_eps=self.eps_,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.sinkhorn_iters,
            )
            return x, residual, residual_mix, post_mix
        return hc_post_pre_norm(
            *input_embdings,
            fn=layer_weight.hc_attn_fn_.weight,
            scale=layer_weight.hc_attn_scale_.weight,
            base=layer_weight.hc_attn_base_.weight,
            norm_weight=layer_weight.attn_norm_.weight,
            streams=self.hc_mult,
            rms_eps=self.eps_,
            norm_eps=self.eps_,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
        )

    def _hc_ffn_in(self, x, residual, residual_mix, post_mix, layer_weight: DeepseekV4TransformerLayerWeight):
        """Finish attention post and run FFN pre + norm."""
        return hc_post_pre_norm(
            layer_output=x,
            residual=residual,
            residual_mix=residual_mix,
            post_mix=post_mix,
            fn=layer_weight.hc_ffn_fn_.weight,
            scale=layer_weight.hc_ffn_scale_.weight,
            base=layer_weight.hc_ffn_base_.weight,
            norm_weight=layer_weight.ffn_norm_.weight,
            streams=self.hc_mult,
            rms_eps=self.eps_,
            norm_eps=self.eps_,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
        )

    def _hc_ffn_out(self, x, residual, residual_mix, post_mix):
        """Defer FFN post until the next pre, or finish it at the last layer."""
        if not self.is_last_layer:
            return x, residual, residual_mix, post_mix
        return hc_post(x, residual, residual_mix, post_mix, self.hc_mult)

    def context_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, residual_mix, post_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.context_attention_forward(x, infer_state, layer_weight)
        x, residual, residual_mix, post_mix = self._hc_ffn_in(x, residual, residual_mix, post_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, residual_mix, post_mix)

    def token_forward(
        self, input_embdings, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        x, residual, residual_mix, post_mix = self._hc_attn_in(input_embdings, layer_weight)
        x = self.token_attention_forward(x, infer_state, layer_weight)
        x, residual, residual_mix, post_mix = self._hc_ffn_in(x, residual, residual_mix, post_mix, layer_weight)
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, residual_mix, post_mix)

    def overlap_tpsp_context_forward(
        self,
        input_embdings,
        input_embdings1,
        infer_state: DeepseekV4InferStateInfo,
        infer_state1: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        experts = layer_weight.experts
        if (
            not self.enable_ep_moe
            or use_mega_moe(experts.quant_method)
            or isinstance(experts.fuse_moe_impl, FuseMoeTritonEP)
        ):
            input_embdings = self.context_forward(input_embdings, infer_state, layer_weight)
            input_embdings1 = self.context_forward(input_embdings1, infer_state1, layer_weight)
            return input_embdings, input_embdings1

        x0, residual0, residual_mix0, post_mix0 = self._hc_attn_in(input_embdings, layer_weight)
        x0 = self.context_attention_forward(x0, infer_state, layer_weight)
        x0, residual0, residual_mix0, post_mix0 = self._hc_ffn_in(x0, residual0, residual_mix0, post_mix0, layer_weight)
        x0 = self._tpsp_allgather(x0.view(-1, self.embed_dim_), infer_state)
        weights0, indices0 = self._select_experts(x0, infer_state, layer_weight)

        indices0 = indices0.to(torch.long)
        qinput0 = experts.quantize_dispatch_input(x0)
        from deep_ep import ElasticBuffer

        dispatch_event0 = ElasticBuffer.capture()
        infer_state1.call_overlap_hook()

        x1, residual1, residual_mix1, post_mix1 = self._hc_attn_in(input_embdings1, layer_weight)
        x1 = self.context_attention_forward(x1, infer_state1, layer_weight)
        x1, residual1, residual_mix1, post_mix1 = self._hc_ffn_in(x1, residual1, residual_mix1, post_mix1, layer_weight)
        x1 = self._tpsp_allgather(x1.view(-1, self.embed_dim_), infer_state1)
        weights1, indices1 = self._select_experts(x1, infer_state1, layer_weight)

        recv_x0, recv_indices0, recv_weights0, recv_count0, handle0, dispatch_hook0 = experts.dispatch(
            qinput0,
            indices0,
            weights0,
            overlap_event=dispatch_event0,
        )
        dispatch_hook0()

        indices1 = indices1.to(torch.long)
        qinput1 = experts.quantize_dispatch_input(x1)

        dispatch_event1 = ElasticBuffer.capture()
        recv_x1, recv_indices1, recv_weights1, recv_count1, handle1, dispatch_hook1 = experts.dispatch(
            qinput1,
            indices1,
            weights1,
            overlap_event=dispatch_event1,
        )

        moe_out0 = experts.prefilled_group_gemm(
            recv_count0,
            handle0.num_unaligned_recv_tokens_per_expert,
            handle0.recv_src_metadata,
            recv_x0,
            recv_indices0,
            recv_weights0,
            hidden_dtype=x0.dtype,
            microbatch_index=infer_state.microbatch_index,
            clamp_limit=self.swiglu_limit,
        )

        combine_event0 = ElasticBuffer.capture()
        routed0, combine_hook0 = experts.combine(moe_out0, handle0, overlap_event=combine_event0)

        dispatch_hook1()
        moe_out1 = experts.prefilled_group_gemm(
            recv_count1,
            handle1.num_unaligned_recv_tokens_per_expert,
            handle1.recv_src_metadata,
            recv_x1,
            recv_indices1,
            recv_weights1,
            hidden_dtype=x1.dtype,
            microbatch_index=infer_state1.microbatch_index,
            clamp_limit=self.swiglu_limit,
        )

        combine_event1 = ElasticBuffer.capture()
        routed1, combine_hook1 = experts.combine(moe_out1, handle1, overlap_event=combine_event1)

        shared0 = self._ffn_tp(x0, infer_state, layer_weight)
        combine_hook0()
        routed0.add_(shared0)
        output0 = self._hc_ffn_out(routed0, residual0, residual_mix0, post_mix0)

        shared1 = self._ffn_tp(x1, infer_state1, layer_weight)
        if self.is_last_layer:
            combine_hook1()
            routed1.add_(shared1)
            output1 = self._hc_ffn_out(routed1, residual1, residual_mix1, post_mix1)
        else:

            def finish():
                combine_hook1()
                routed1.add_(shared1)

            infer_state1.hook = finish
            output1 = routed1, residual1, residual_mix1, post_mix1
        return output0, output1

    def overlap_tpsp_token_forward(
        self,
        input_embdings,
        input_embdings1,
        infer_state: DeepseekV4InferStateInfo,
        infer_state1: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        experts = layer_weight.experts
        if (
            not self.enable_ep_moe
            or use_mega_moe(experts.quant_method)
            or isinstance(experts.fuse_moe_impl, FuseMoeTritonEP)
        ):
            input_embdings = self.token_forward(input_embdings, infer_state, layer_weight)
            input_embdings1 = self.token_forward(input_embdings1, infer_state1, layer_weight)
            return input_embdings, input_embdings1

        x0, residual0, residual_mix0, post_mix0 = self._hc_attn_in(input_embdings, layer_weight)
        x0 = self.token_attention_forward(x0, infer_state, layer_weight)
        x0, residual0, residual_mix0, post_mix0 = self._hc_ffn_in(x0, residual0, residual_mix0, post_mix0, layer_weight)
        x0 = self._tpsp_allgather(x0.view(-1, self.embed_dim_), infer_state)
        weights0, indices0 = self._select_experts(x0, infer_state, layer_weight)
        infer_state1.call_overlap_hook()

        shared0 = self._ffn_tp(x0, infer_state, layer_weight)
        recv_x0, masked_m0, indices0, weights0, handle0, dispatch_hook0 = experts.low_latency_dispatch_with_topk(
            x0, indices0, weights0
        )

        x1, residual1, residual_mix1, post_mix1 = self._hc_attn_in(input_embdings1, layer_weight)
        x1 = self.token_attention_forward(x1, infer_state1, layer_weight)
        x1, residual1, residual_mix1, post_mix1 = self._hc_ffn_in(x1, residual1, residual_mix1, post_mix1, layer_weight)
        x1 = self._tpsp_allgather(x1.view(-1, self.embed_dim_), infer_state1)
        weights1, indices1 = self._select_experts(x1, infer_state1, layer_weight)
        dispatch_hook0()

        shared1 = self._ffn_tp(x1, infer_state1, layer_weight)
        recv_x1, masked_m1, indices1, weights1, handle1, dispatch_hook1 = experts.low_latency_dispatch_with_topk(
            x1, indices1, weights1
        )

        expected_m = triton.cdiv(
            x0.shape[0] * get_global_world_size() * self.num_experts_per_tok,
            layer_weight.n_routed_experts,
        )
        moe_out0 = experts.masked_group_gemm(
            recv_x0,
            masked_m0,
            x0.dtype,
            expected_m,
            clamp_limit=self.swiglu_limit,
        )

        dispatch_hook1()
        routed0, combine_hook0 = experts.low_latency_combine(moe_out0, indices0, weights0, handle0)

        moe_out1 = experts.masked_group_gemm(
            recv_x1,
            masked_m1,
            x1.dtype,
            expected_m,
            clamp_limit=self.swiglu_limit,
        )

        combine_hook0()
        routed0.add_(shared0)
        output0 = self._hc_ffn_out(routed0, residual0, residual_mix0, post_mix0)

        routed1, combine_hook1 = experts.low_latency_combine(moe_out1, indices1, weights1, handle1)
        if self.is_last_layer:
            combine_hook1()
            routed1.add_(shared1)
            output1 = self._hc_ffn_out(routed1, residual1, residual_mix1, post_mix1)
        else:

            def finish():
                combine_hook1()
                routed1.add_(shared1)

            infer_state1.hook = finish
            output1 = routed1, residual1, residual_mix1, post_mix1
        return output0, output1

    # ------------------------------------------------------------------ shared projections / cache
    def _select_rope(self, infer_state: DeepseekV4InferStateInfo):
        if self.compress_ratio:
            return infer_state.position_cos_compress, infer_state.position_sin_compress
        return infer_state.position_cos_sliding, infer_state.position_sin_sliding

    def _norm_rope_and_cache_kv(self, kv, infer_state, layer_weight):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_kv_norm_rope_to_swa

        req_manager = infer_state.req_manager
        att_state = infer_state.prefill_att_state if infer_state.is_prefill else infer_state.decode_att_state
        swa, _ = att_state.backend.get_att_input_params(infer_state, self.layer_num_)
        # 一次完成 RMSNorm、RoPE、FP8 打包和 SWA 写入。
        fused_kv_norm_rope_to_swa(
            kv=kv,
            kv_norm_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            position_ids=infer_state.position_ids,
            swa_write_slots=infer_state.dsv4_swa_write_slots,
            swa_runtime_write_slots=infer_state.dsv4_swa_runtime_write_slots,
            swa_kv_buffer=swa,
            swa_page_token_num=req_manager.cache_config.swa_page_token_num,
            req_to_swa=req_manager.req_to_swa[self.layer_num_],
        )

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4TransformerLayerWeight,
    ):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_norm_rope

        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        T = input.shape[0]

        qkv = layer_weight.wq_a_wkv_.mm(input)
        qa = layer_weight.q_norm_(qkv[:, : -self.head_dim_], eps=self.eps_)
        q_in = layer_weight.wq_b_.mm(qa).view(T, self.tp_q_head_num_, self.head_dim_)

        if infer_state.is_prefill:
            q = infer_state.dsv4_workspace.get_flashmla_prefill_q(T)
        else:
            q = self.alloc_tensor((T, self.flashmla_q_head_num_, self.head_dim_), dtype=q_in.dtype, device=q_in.device)
            q[:, self.tp_q_head_num_ :, :].zero_()
        fused_q_norm_rope(q_in, q[:, : self.tp_q_head_num_, :], self.eps_, self.freqs_cis, infer_state.position_ids)
        self._norm_rope_and_cache_kv(qkv[:, -self.head_dim_ :], infer_state, layer_weight)
        return q, qa, input

    def _get_o(self, o, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        # o: [T, tp_q_head_num_, head_dim_] after inverse rope -> grouped low-rank O -> [T, embed_dim_]
        position_cos, position_sin = self._select_rope(infer_state)
        inverse_rotary_emb(o[..., -self.qk_rope_head_dim :], position_cos, position_sin)
        T = o.shape[0]
        if layer_weight.o_proj_use_mm:
            o = layer_weight.wo_a_.mm(o.reshape(T, -1))  # [T, o_lora]
        else:
            o = o.reshape(T, self.tp_groups, -1).transpose(0, 1)  # [groups, T, per_group_in]
            o = layer_weight.wo_a_.bmm(o).transpose(0, 1).reshape(T, -1)  # [T, groups*o_lora]
        o = layer_weight.wo_b_.mm(o)
        return self._tpsp_reduce(input=o, infer_state=infer_state)

    # ------------------------------------------------------------------ attention (prefill)
    def context_attention_forward(
        self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight
    ):
        # _get_qkv writes the chunk's packed latent into the swa pool (fused kernel) before
        # attention reads it back via request-owned SWA indices (this custom forward bypasses the
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
            o = self.alloc_tensor((q.shape[0], q.shape[1], self.head_dim_), dtype=q.dtype, device=q.device)
            _o = tensor_to_no_ref_tensor(o)

            def att_func(new_infer_state: DeepseekV4InferStateInfo):
                self._context_attention_kernel(_q, _q_lora, _x, new_infer_state, layer_weight, out=_o)
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=att_func, after_graph=pre_capture_graph)
            return o[:, : self.tp_q_head_num_]

        return self._context_attention_kernel(q, q_lora, x, infer_state, layer_weight)

    def _compress_and_index(self, q_lora, x, infer_state: DeepseekV4InferStateInfo, layer_weight):
        cos_table, sin_table = self.cos_compress_table, self.sin_compress_table
        aux_stream = self.dsv4_prefill_aux_stream
        if (
            infer_state.is_prefill
            and self.compress_ratio == 4
            and aux_stream is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            main_stream = torch.cuda.current_stream()
            aux_stream.wait_stream(main_stream)  # aux waits for x / q_lora produced on main
            with torch.cuda.stream(aux_stream):
                # x / q_lora are main-allocated and read here -> record so the allocator won't reuse them.
                x.record_stream(aux_stream)
                q_lora.record_stream(aux_stream)
                self.index_infer.write_indexer_k(
                    x,
                    infer_state,
                    layer_weight,
                    cos_table,
                    sin_table,
                    use_custom_tensor_manager=False,
                    c4_aux_workspace=infer_state.dsv4_workspace.c4_prefill_aux,
                )
                meta = self.index_infer.build_metadata(
                    x,
                    q_lora,
                    infer_state,
                    layer_weight,
                    use_custom_tensor_manager=False,
                    c4_aux_workspace=infer_state.dsv4_workspace.c4_prefill_aux,
                )
            self.compressor.compress(x, infer_state, layer_weight, cos_table, sin_table)
            main_stream.wait_stream(aux_stream)  # join before prefill_att reads the indices / latent KV
            # C4 indices and lengths are allocated on aux and consumed on main.
            meta["extra_indices"].record_stream(main_stream)
            meta["extra_lengths"].record_stream(main_stream)
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
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "attn_sink": layer_weight.attn_sink_.weight,
                **meta,
            },
        )
        return infer_state.prefill_att_state.prefill_att(
            q=q,
            k=infer_state.prefill_att_state.backend.get_att_input_params(infer_state, self.layer_num_),
            v=None,
            att_control=att_control,
            alloc_func=self.alloc_tensor,
            out=out,
        )

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
        meta = self._compress_and_index(q_lora, x, infer_state, layer_weight)
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "compress_ratio": self.compress_ratio,
                "head_dim_v": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "attn_sink": layer_weight.attn_sink_.weight,
                **meta,
            },
        )
        return infer_state.decode_att_state.decode_att(
            q=q,
            k=infer_state.decode_att_state.backend.get_att_input_params(infer_state, self.layer_num_),
            v=None,
            att_control=att_control,
        )

    # ------------------------------------------------------------------ moe
    def _ffn_tp(self, input, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        input = input.view(-1, self.embed_dim_)
        gate_up = layer_weight.gate_up_proj.mm(input)
        shared = self.alloc_tensor((input.size(0), gate_up.size(1) // 2), input.dtype)
        silu_and_mul_fwd(
            gate_up,
            shared,
            alpha=1.0,
            limit=self.swiglu_limit,
            clamp_up_add_one=False,
        )
        del gate_up
        return layer_weight.down_proj.mm(shared)

    def _ffn(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        x = x.view(-1, self.embed_dim_)
        x = self._tpsp_allgather(input=x, infer_state=infer_state)

        topk_weights, topk_indices = self._select_experts(x, infer_state, layer_weight)
        # Routed experts can overwrite x in place, so compute the shared expert first.
        shared = self._ffn_tp(input=x, infer_state=infer_state, layer_weight=layer_weight)
        routed = layer_weight.experts.experts_with_topk(
            input_tensor=x,
            topk_weights=topk_weights,
            topk_ids=topk_indices,
            is_prefill=infer_state.is_prefill,
            infer_state=infer_state,
            clamp_limit=self.swiglu_limit,
        )
        out = routed + shared
        if self.enable_ep_moe:
            return out
        return self._tpsp_reduce(input=out, infer_state=infer_state)

    def _select_experts(self, x, infer_state: DeepseekV4InferStateInfo, layer_weight: DeepseekV4TransformerLayerWeight):
        num_tokens = x.shape[0]
        logits = self.alloc_tensor((num_tokens, layer_weight.n_routed_experts), dtype=torch.float32, device=x.device)
        layer_weight.gate_weight_.mm(x, out=logits)
        correction_bias = None
        bias_vl = None
        input_ids = None
        tid2eid = None
        image_token_start = 0
        indices_dtype = torch.int64
        if self.is_hash:
            tid2eid = layer_weight.gate_tid2eid_.weight
            indices_dtype = tid2eid.dtype
            input_ids = infer_state.input_ids.to(dtype=indices_dtype)
        else:
            correction_bias = layer_weight.gate_bias_.weight
        if self.has_vision and infer_state.is_prefill:
            bias_vl = layer_weight.gate_bias_vl_.weight
            image_token_start = self.vocab_size
            if input_ids is None:
                input_ids = infer_state.input_ids

        topk_weights = self.alloc_tensor(
            (num_tokens, self.num_experts_per_tok), dtype=torch.float32, device=logits.device
        )
        topk_indices = self.alloc_tensor(
            (num_tokens, self.num_experts_per_tok), dtype=indices_dtype, device=logits.device
        )
        topk_softplus_sqrt(
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            logits=logits,
            routed_scaling_factor=self.routed_scaling_factor,
            correction_bias=correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            bias_vl=bias_vl,
            image_token_start=image_token_start,
        )
        return topk_weights, topk_indices
