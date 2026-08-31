import os

import torch
from typing import Any
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import (
    Deepseek2TransformerLayerInfer,
)
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import (
    Deepseek3_2TransformerLayerWeight,
)
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import (
    destindex_copy_indexer_ks,
)
from lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks import (
    extract_indexer_ks,
    extract_indexer_ks_dynamic,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.distributed import all_gather_into_tensor


class Deepseek3_2TransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self.index_topk = network_config["index_topk"]
        super().__init__(layer_num, network_config)

        self.indexer = NsaInfer(
            layer_idx=self.layer_num_,
            network_config=self.network_config_,
            tp_world_size=self.tp_world_size_,
        )
        return

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_unbalance_get(data=input)

        q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

        infer_state.get_topk_indices_params = {
            "hidden_states": input,
            "q_lora": q,
        }

        q = layer_weight.q_b_proj_.mm(q)
        cache_kv = cache_kv.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        # Model-specific q projection (uses layer weights)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # 计算 topk indices
        att_state = infer_state.prefill_att_state
        topk_mem_indices, topk_indices = self.indexer._get_indices(
            hidden_states=infer_state.get_topk_indices_params["hidden_states"],
            q_lora=infer_state.get_topk_indices_params["q_lora"],
            infer_state=infer_state,
            att_state=att_state,
            layer_weight=layer_weight,
        )
        del infer_state.get_topk_indices_params

        # Use NSA backend for attention computation
        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "topk_mem_indices": topk_mem_indices,
                "topk_indices": topk_indices,
                "prefill_cache_kv": kv,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
            },
        )

        mla_out = infer_state.prefill_att_state.prefill_att(
            q=q_all,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )
        return mla_out

    def _token_attention_kernel(
        self,
        q,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ):
        # Model-specific q projection (uses layer weights)
        q_nope, q_rope = (
            q[:, :, : -self.qk_rope_head_dim],
            q[:, :, -self.qk_rope_head_dim :],
        )
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        # 计算 topk mem indices
        att_state = infer_state.decode_att_state
        topk_mem_indices, _ = self.indexer._get_indices(
            hidden_states=infer_state.get_topk_indices_params["hidden_states"],
            q_lora=infer_state.get_topk_indices_params["q_lora"],
            infer_state=infer_state,
            att_state=att_state,
            layer_weight=layer_weight,
        )
        del infer_state.get_topk_indices_params

        # Use NSA backend for attention computation
        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "layer_index": self.layer_num_,
                "topk_mem_indices": topk_mem_indices,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
            },
        )

        o_tensor = infer_state.decode_att_state.decode_att(
            q=(q_nope, q_rope),
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )
        return o_tensor


class NsaInfer:
    _MQA_LOGITS_BYTES_PER_ELEM = 4
    _MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
    _MQA_LOGITS_FREE_MEM_FRACTION = 0.2
    _FLASHMLA_SPARSE_TOPK_ALIGNMENT = 128
    _mqa_logits_budget_bytes = {}

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int):
        super().__init__()
        self.layer_idx_ = layer_idx
        self.network_config_ = network_config
        self.index_topk = network_config["index_topk"]
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.index_head_dim = network_config["index_head_dim"]
        self.eps = network_config["rms_norm_eps"]
        self.block_size = network_config["quantization_config"]["weight_block_size"][0]
        self.scale_fmt = network_config["quantization_config"]["scale_fmt"]
        self.softmax_scale = (self.index_head_dim) ** (-0.5)
        self.index_n_heads = network_config["index_n_heads"]
        self.index_n_heads_scale = (self.index_n_heads ** -0.5) * self.softmax_scale
        self.tp_world_size_ = tp_world_size
        self.tp_index_n_heads = self.index_n_heads // self.tp_world_size_
        self.index_kpool = network_config.get("index_kpool", 1)
        self.index_kpool_compress = network_config.get("index_kpool_compress", False)
        self.enable_kpool_decode_fastpath = os.getenv("LIGHTLLM_ENABLE_KPOOL_DECODE_FASTPATH", "0").upper() in {
            "1",
            "ON",
            "TRUE",
        }
        self._kpool_tail_k = None
        self._kpool_tail_score = None
        # Most NSA models only instantiate the target model, so their decode
        # layout follows the process-wide MTP setting.  A model that reuses an
        # NSA layer as a recurrent drafter can override this with its own
        # physical decode width (normally zero extra rows).
        self.decode_mtp_step = None

    def _get_indices(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        att_state: Any,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ):
        q_k = self._get_q_k_bf16(hidden_states, q_lora, infer_state, layer_weight)
        if len(q_k) == 2:
            q, k = q_k
            raw_k = None
        else:
            q, k, raw_k = q_k

        if self.tp_world_size_ > 1:
            q_merge = torch.empty(
                size=(self.tp_world_size_ * q.numel(),),
                dtype=q.dtype,
                device=q.device,
            )
            all_gather_into_tensor(
                output_=q_merge,
                input_=q.view(-1),
                group=infer_state.dist_group,
                async_op=False,
            )
            q = (
                q_merge.view(self.tp_world_size_, q.shape[0], self.tp_index_n_heads, q.shape[2])
                .transpose(0, 1)
                .contiguous()
                .view(q.shape[0], self.index_n_heads, q.shape[2])
            )

        q_fp8, q_scale = self._quantize_indexer_activation(q)
        k_fp8, k_scale = self._quantize_indexer_activation(k)

        indexer_k_buffer = infer_state.mem_manager.get_indexer_k_buffer(self.layer_idx_)
        destindex_copy_indexer_ks(
            K_fp8=k_fp8,
            K_scale=k_scale,
            DestLoc=infer_state.mem_index,
            O_buffer=indexer_k_buffer,
        )

        weights = self._scale_indexer_weights(layer_weight.weights_proj_.mm(hidden_states), q_scale)

        ks = att_state.ks
        ke = att_state.ke
        lengths = att_state.lengths

        use_kpool_prefill = (
            infer_state.is_prefill
            and self.index_kpool > 1
            and self.index_kpool_compress
            and raw_k is not None
            and infer_state.kpool_prefill_aligned
            and infer_state.mem_index.shape[0] == q_fp8.shape[0]
        )
        use_kpool_decode = (
            not infer_state.is_prefill
            and self.enable_kpool_decode_fastpath
            and infer_state.kpool_decode_aligned
            and raw_k is not None
            and get_env_start_args().mtp_mode is None
        )
        use_kpool = use_kpool_prefill or use_kpool_decode
        if use_kpool_prefill:
            (k_fp8_, k_scale_, score_ks, score_ke, score_lengths,) = self._prepare_kpool_scoring(
                raw_k=raw_k,
                hidden_states=hidden_states,
                q_lora=q_lora,
                infer_state=infer_state,
                layer_weight=layer_weight,
                indexer_k_buffer=indexer_k_buffer,
                ragged_mem_index=att_state.ragged_mem_index,
                ks=ks,
                lengths=lengths,
            )
            use_kpool = k_fp8_ is not None
        elif use_kpool_decode:
            (k_fp8_, k_scale_, score_ks, score_ke, score_lengths,) = self._prepare_kpool_decode_scoring(
                raw_k=raw_k,
                hidden_states=hidden_states,
                q_lora=q_lora,
                infer_state=infer_state,
                layer_weight=layer_weight,
                indexer_k_buffer=indexer_k_buffer,
                lengths=lengths,
            )
            use_kpool = k_fp8_ is not None

        if use_kpool:
            mtp_step = 0
        elif infer_state.is_prefill:
            mtp_step = 0
        else:
            mtp_step = get_env_start_args().mtp_step if self.decode_mtp_step is None else self.decode_mtp_step
        # LightSpec compacts each request to a variable number of contiguous
        # verify rows. Its sparse-index K packing must follow request boundaries
        # instead of assuming the fixed process-wide MTP width.
        use_dynamic_layout = not infer_state.is_prefill and mtp_step > 0 and get_env_start_args().mtp_dynamic_verify
        if use_kpool:
            pass
        elif use_dynamic_layout:
            k_fp8_, k_scale_ = extract_indexer_ks_dynamic(
                I_buffer=indexer_k_buffer,
                b_seq_len=infer_state.b_seq_len,
                b_req_idx=infer_state.b_req_idx,
                b_mtp_index=infer_state.b_mtp_index,
                req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
                max_kv_seq_len=infer_state.max_kv_seq_len,
                max_request_num=infer_state.req_manager.max_request_num,
            )
        else:
            k_fp8_, k_scale_ = extract_indexer_ks(
                I_buffer=indexer_k_buffer,
                b_seq_len=infer_state.b_seq_len,
                b_req_idx=infer_state.b_req_idx,
                req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
                out_token_num=infer_state.b_seq_len.shape[0] * infer_state.max_kv_seq_len,
                max_kv_seq_len=infer_state.max_kv_seq_len,
                mtp_step=mtp_step,
            )
        if not use_kpool:
            score_ks, score_ke, score_lengths = ks, ke, lengths

        import deep_gemm

        from sgl_kernel import fast_topk_v2

        query_token_num = q_fp8.shape[0]
        kv_token_num = k_fp8_.shape[0]
        query_chunk_size = self._get_mqa_logits_chunk_size(
            query_token_num=query_token_num,
            kv_token_num=kv_token_num,
            device=q_fp8.device,
        )

        if use_kpool:
            output_topk = (
                (self.index_topk + self.index_kpool - 1 + self._FLASHMLA_SPARSE_TOPK_ALIGNMENT - 1)
                // self._FLASHMLA_SPARSE_TOPK_ALIGNMENT
                * self._FLASHMLA_SPARSE_TOPK_ALIGNMENT
            )
            # K-pool appends at most pool_size - 1 always-selected tail
            # tokens. Allocate the FlashMLA-aligned result once and leave the
            # alignment gap masked, avoiding two full-size top-k buffers.
            b_topk_index = torch.full(
                (query_token_num, output_topk),
                -1,
                dtype=torch.int32,
                device=q_fp8.device,
            )
        else:
            b_topk_index = torch.empty(
                (query_token_num, self.index_topk),
                dtype=torch.int32,
                device=q_fp8.device,
            )
        for start in range(0, query_token_num, query_chunk_size):
            end = min(start + query_chunk_size, query_token_num)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end],
                (k_fp8_, k_scale_),
                weights[start:end],
                score_ks[start:end],
                score_ke[start:end],
                # fast top-k already masks columns past each row's valid
                # length, while DeepGEMM's small-M path rejects clean_logits.
                clean_logits=False,
                max_seqlen_k=kv_token_num,
            )
            if use_kpool:
                from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
                    topk_from_pooled_history_logits,
                )

                topk_index = topk_from_pooled_history_logits(
                    logits=logits,
                    group_lengths=score_lengths[start:end],
                    pool_size=self.index_kpool,
                    topk=self.index_topk,
                    seq_lens=lengths[start:end],
                )
            else:
                topk_index = fast_topk_v2(
                    score=logits,
                    lengths=score_lengths[start:end],
                    topk=self.index_topk,
                )
            b_topk_index[start:end, : topk_index.shape[1]].copy_(topk_index)
            # The long-prefill score matrix can be tens of GiB. Release each
            # chunk before computing the next one.
            del logits, topk_index
        # 将 topk index 转化为 mem index
        from ..triton_kernel.topk_index_to_mem_index import (
            trans_topk_index_to_mem_index,
        )

        b_topk_mem_index = trans_topk_index_to_mem_index(
            topk_index=b_topk_index,
            ragged_start_index=ks,
            ragged_mem_index=att_state.ragged_mem_index,
        )

        return b_topk_mem_index, b_topk_index

    def _quantize_indexer_activation(self, value: torch.Tensor):
        return act_quant(value, self.block_size, self.scale_fmt)

    def _scale_indexer_weights(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        return (weights.mul(self.index_n_heads_scale).unsqueeze(-1).mul(q_scale)).squeeze(-1)

    def _prepare_kpool_scoring(
        self,
        raw_k,
        hidden_states,
        q_lora,
        infer_state,
        layer_weight,
        indexer_k_buffer,
        ragged_mem_index,
        ks,
        lengths,
    ):
        pool_size = self.index_kpool
        query_token_num = raw_k.shape[0]
        if not infer_state.kpool_prefill_aligned:
            return None, None, None, None, None
        # Without decode K-pool enabled the zero-prefix path is intentionally
        # transient, so no pooled history exists for a later chunk.
        if infer_state.max_cache_len > 0 and not self.enable_kpool_decode_fastpath:
            return None, None, None, None, None

        gate_score = layer_weight.index_kpool_compress_gate.mm(hidden_states.to(q_lora.dtype))
        closed_pool_num = query_token_num // pool_size
        compressed_k = None
        compressed_scale = None
        if closed_pool_num:
            closed_token_num = closed_pool_num * pool_size
            slot_k = raw_k[:closed_token_num].view(closed_pool_num, pool_size, self.index_head_dim)
            slot_score = gate_score[:closed_token_num].view(closed_pool_num, pool_size, self.index_head_dim)
            write_locs = infer_state.mem_index[:closed_token_num].view(closed_pool_num, pool_size)[:, -1]
            compressed_k, compressed_scale = self._compress_kpool_keys(
                slot_k=slot_k,
                slot_score=slot_score,
                write_locs=write_locs,
                layer_weight=layer_weight,
                output_buffer=indexer_k_buffer,
                persist=self.enable_kpool_decode_fastpath,
            )

        # The common serving path has no prefix and each request fits in this
        # aligned chunk.  DeepGEMM can consume the freshly compressed pools
        # directly; allocating and gathering a second max-token-sized cache
        # would waste several GiB for GLM-5.3 TP8 and can make c64 OOM.
        if infer_state.max_cache_len == 0:
            if compressed_k is None:
                return None, None, None, None, None
            pool_lengths = torch.div(lengths, pool_size, rounding_mode="floor").to(torch.int32)
            score_ks = torch.div(ks, pool_size, rounding_mode="floor").to(torch.int32)
            score_ke = score_ks + pool_lengths
            return compressed_k, compressed_scale, score_ks, score_ke, pool_lengths

        pooled_token_num = infer_state.total_token_num // pool_size
        if pooled_token_num == 0:
            return None, None, None, None, None
        # Every request boundary is pool-aligned, so selecting each pool's
        # final token from the ragged request-major layout preserves exactly
        # the packed-K order expected by DeepGEMM's per-row ks/ke ranges.
        pooled_ragged_positions = torch.arange(
            pool_size - 1,
            infer_state.total_token_num,
            pool_size,
            dtype=torch.int64,
            device=raw_k.device,
        )
        mem_indices = ragged_mem_index[pooled_ragged_positions].to(torch.int64)
        packed_k = indexer_k_buffer[mem_indices, 0]
        k_fp8 = packed_k[:, : self.index_head_dim].contiguous().view(torch.float8_e4m3fn)
        k_scale = packed_k[:, self.index_head_dim : self.index_head_dim + 4].contiguous().view(torch.float32).view(-1)

        pool_lengths = torch.div(lengths, pool_size, rounding_mode="floor").to(torch.int32)
        score_ks = torch.div(ks, pool_size, rounding_mode="floor").to(torch.int32)
        score_ke = score_ks + pool_lengths
        return k_fp8, k_scale, score_ks, score_ke, pool_lengths

    def _prepare_kpool_decode_scoring(
        self,
        raw_k,
        hidden_states,
        q_lora,
        infer_state,
        layer_weight,
        indexer_k_buffer,
        lengths,
    ):
        """Update one-token K-pool tails and gather pooled decode history.

        This fast path deliberately targets plain decode.  MTP verify has
        multiple speculative rows per request and needs acceptance-aware tail
        rollback, so it remains on the regular indexer path.
        """

        pool_size = self.index_kpool
        batch_size = raw_k.shape[0]
        if batch_size == 0:
            return None, None, None, None, None

        if self._kpool_tail_k is None:
            tail_shape = (
                infer_state.req_manager.max_request_num + 1,
                pool_size,
                self.index_head_dim,
            )
            self._kpool_tail_k = torch.empty(tail_shape, dtype=torch.bfloat16, device=raw_k.device)
            self._kpool_tail_score = torch.empty_like(self._kpool_tail_k)

        req_idx = infer_state.b_req_idx.to(torch.int64)
        positions = infer_state.b_seq_len.to(torch.int64) - 1
        tail_slots = torch.remainder(positions, pool_size)
        gate_score = layer_weight.index_kpool_compress_gate.mm(hidden_states.to(q_lora.dtype))
        self._kpool_tail_k[req_idx, tail_slots] = raw_k
        self._kpool_tail_score[req_idx, tail_slots] = gate_score

        # Compress every row's current tail.  Only pool-end token locations are
        # gathered as history, so writes at intermediate token locations are
        # harmless and avoid a dynamic nonzero/host synchronization.
        self._compress_kpool_keys(
            slot_k=self._kpool_tail_k[req_idx],
            slot_score=self._kpool_tail_score[req_idx],
            write_locs=infer_state.mem_index,
            layer_weight=layer_weight,
            output_buffer=indexer_k_buffer,
            persist=True,
        )

        max_pool_len = infer_state.max_kv_seq_len // pool_size
        if max_pool_len == 0:
            return None, None, None, None, None
        pool_lengths = torch.div(lengths, pool_size, rounding_mode="floor").to(torch.int32)
        endpoint_positions = (
            torch.arange(max_pool_len, dtype=torch.int64, device=raw_k.device) * pool_size + pool_size - 1
        )
        last_endpoint = torch.clamp(pool_lengths.to(torch.int64) * pool_size - 1, min=0)
        safe_positions = torch.minimum(endpoint_positions.unsqueeze(0), last_endpoint.unsqueeze(1))
        mem_indices = infer_state.req_manager.req_to_token_indexs[req_idx.unsqueeze(1), safe_positions].to(torch.int64)
        packed_k = indexer_k_buffer[mem_indices.reshape(-1), 0]
        k_fp8 = packed_k[:, : self.index_head_dim].contiguous().view(torch.float8_e4m3fn)
        k_scale = packed_k[:, self.index_head_dim : self.index_head_dim + 4].contiguous().view(torch.float32).view(-1)
        score_ks = torch.arange(batch_size, dtype=torch.int32, device=raw_k.device) * max_pool_len
        score_ke = score_ks + pool_lengths
        return k_fp8, k_scale, score_ks, score_ke, pool_lengths

    def _compress_kpool_keys(
        self,
        slot_k,
        slot_score,
        write_locs,
        layer_weight,
        output_buffer,
        persist,
    ):
        from types import SimpleNamespace

        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_softmax_rotate_write_cache,
        )

        compressed_k, compressed_scale = kpool_softmax_rotate_write_cache(
            pool=SimpleNamespace(page_size=64, index_head_dim=self.index_head_dim),
            buf=output_buffer,
            slot_k=slot_k,
            slot_score=slot_score,
            ape=layer_weight.index_kpool_compress_ape.weight,
            loc=write_locs.to(torch.int64),
            round_scale=self.scale_fmt is not None,
            return_compressed=True,
            write_cache=False,
        )
        if persist:
            destindex_copy_indexer_ks(
                K_fp8=compressed_k,
                K_scale=compressed_scale,
                DestLoc=write_locs,
                O_buffer=output_buffer,
            )
        return compressed_k, compressed_scale

    @classmethod
    def _get_mqa_logits_chunk_size(cls, query_token_num: int, kv_token_num: int, device: torch.device) -> int:
        score_element_num = query_token_num * kv_token_num
        if score_element_num < cls._MQA_LOGITS_STATIC_SKIP_ELEMS:
            return query_token_num

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        logits_budget_bytes = cls._mqa_logits_budget_bytes.get(device_index)
        if logits_budget_bytes is None:
            free_memory_bytes, _ = torch.cuda.mem_get_info(device_index)
            free_mem_fraction = float(
                os.getenv(
                    "LIGHTLLM_MQA_LOGITS_FREE_MEM_FRACTION",
                    cls._MQA_LOGITS_FREE_MEM_FRACTION,
                )
            )
            free_mem_fraction = min(max(free_mem_fraction, 0.01), 0.9)
            logits_budget_bytes = max(1, int(free_memory_bytes * free_mem_fraction))
            cls._mqa_logits_budget_bytes[device_index] = logits_budget_bytes

        bytes_per_query = kv_token_num * cls._MQA_LOGITS_BYTES_PER_ELEM
        return max(1, min(query_token_num, logits_budget_bytes // bytes_per_query))

    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import (
            hadamard_transform,
        )

        hidden_size = x.size(-1)
        assert (hidden_size & (hidden_size - 1)) == 0, "Hidden size must be a power of 2 for Hadamard transform."
        return hadamard_transform(x, scale=hidden_size ** -0.5)

    def _get_q_k_bf16(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ):
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.tp_index_n_heads, self.index_head_dim)
        k = layer_weight.wk_proj_.mm(hidden_states)

        k = layer_weight.k_norm_(k, eps=self.eps)

        # 为什么 indexer 和主模型用的q k 的 rotary的排布方式不一样，这不是脱裤子放屁麻。
        from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

        rotary_emb_fwd(
            q[:, :, : self.qk_rope_head_dim],
            k[:, None, : self.qk_rope_head_dim],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        q = self._rotate_activation(q)
        k = self._rotate_activation(k)
        return q, k
