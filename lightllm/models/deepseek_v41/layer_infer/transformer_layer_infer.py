import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.mhc.pre_norm import hc_pre_norm_shifted
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
from lightllm.common.basemodel.triton_kernel.mhc import hc_post
from lightllm.models.deepseek_v4.triton_kernel.cache_io import (
    get_compressed_kv_buffer,
    get_indexer_k_buffer,
)
from ..engram import engram_forward
from ..infer_struct import DeepseekV41InferStateInfo
from ..layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
from ..triton_kernel.attention import (
    rotary_quantize_fp4,
    pool_pairs,
    destindex_copy_kv_dsv41,
    index_logits,
)
from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import destindex_copy_indexer_k_dsv4


class DeepseekV41TransformerLayerInfer(DeepseekV4TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.is_kv_source = layer_num in network_config["kv_source_layer_ids"]
        self.decoder_start_layer = max(network_config["kv_source_layer_ids"]) + 1
        self.is_index_source = layer_num in network_config["index_source_layer_ids"]
        self.is_candidate_source = layer_num == network_config["candidate_source_layer_id"]
        self.uses_candidates = layer_num > network_config["candidate_source_layer_id"]
        self.index_topk = network_config["index_topk"]
        self.candidate_blocks = network_config["candidate_topk_blocks"]
        self.engram_index = (
            network_config["engram_layer_ids"].index(layer_num)
            if layer_num in network_config["engram_layer_ids"]
            else None
        )

    def _hc_attn_in(self, input_embdings, infer_state, layer_weight):
        if isinstance(input_embdings, tuple):
            streams, pre_mix = input_embdings
        else:
            streams = input_embdings
            pre_mix = streams.new_zeros((streams.shape[0], self.hc_mult), dtype=torch.float32)
            pre_mix[:, 0] = 1
        if self.engram_index is not None:
            streams = engram_forward(
                streams.view(-1, self.hc_mult, self.embed_dim_),
                infer_state.engram_hashes[:, self.engram_index],
                layer_weight,
                self.eps_,
            ).flatten(1)
        x, res_mix, post_mix, attn_pre = hc_pre_norm_shifted(
            streams,
            pre_mix,
            layer_weight.hc_attn_fn_.weight,
            layer_weight.hc_attn_scale_.weight,
            layer_weight.hc_attn_base_.weight,
            layer_weight.attn_norm_.weight,
            self.eps_,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        return x, streams, res_mix, post_mix, attn_pre

    def _hc_ffn_in(self, x, residual, res_mix, post_mix, attn_pre, layer_weight):
        streams = hc_post(x, residual, res_mix, post_mix, streams=self.hc_mult)
        x, res_mix, post_mix, ffn_pre = hc_pre_norm_shifted(
            streams,
            attn_pre,
            layer_weight.hc_ffn_fn_.weight,
            layer_weight.hc_ffn_scale_.weight,
            layer_weight.hc_ffn_base_.weight,
            layer_weight.ffn_norm_.weight,
            self.eps_,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        return x, streams, res_mix, post_mix, ffn_pre

    def _hc_ffn_out(self, x, residual, res_mix, post_mix, ffn_pre):
        streams = hc_post(x, residual, res_mix, post_mix, streams=self.hc_mult)
        return streams, ffn_pre

    def context_forward(
        self, input_embdings, infer_state: DeepseekV41InferStateInfo, layer_weight: DeepseekV41TransformerLayerWeight
    ):
        if not infer_state.return_all_prompt_logics:
            if self.layer_num_ == self.decoder_start_layer:
                rows = infer_state.init_decoder_state()
                input_embdings = tuple(value[rows] for value in input_embdings)
            if infer_state.decoder_state is not None:
                infer_state = infer_state.decoder_state
        x, residual, res_mix, post_mix, attn_pre = self._hc_attn_in(input_embdings, infer_state, layer_weight)
        x = self.context_attention_forward(x, infer_state, layer_weight)
        x, residual, res_mix, post_mix, ffn_pre = self._hc_ffn_in(
            x, residual, res_mix, post_mix, attn_pre, layer_weight
        )
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, res_mix, post_mix, ffn_pre)

    def token_forward(
        self, input_embdings, infer_state: DeepseekV41InferStateInfo, layer_weight: DeepseekV41TransformerLayerWeight
    ):
        x, residual, res_mix, post_mix, attn_pre = self._hc_attn_in(input_embdings, infer_state, layer_weight)
        x = self.token_attention_forward(x, infer_state, layer_weight)
        x, residual, res_mix, post_mix, ffn_pre = self._hc_ffn_in(
            x, residual, res_mix, post_mix, attn_pre, layer_weight
        )
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, res_mix, post_mix, ffn_pre)

    def _compress_and_index(self, q_lora, x, infer_state: DeepseekV41InferStateInfo, layer_weight):
        if self.is_kv_source:
            self._write_global_kv(x, infer_state, layer_weight)
        if self.is_index_source:
            infer_state.v41_sparse_selection = self._build_index_metadata(x, q_lora, infer_state, layer_weight)
        meta = {
            "swa_indices": infer_state.dsv4_swa_indices[:, None, :],
            "swa_lengths": infer_state.dsv4_swa_lengths,
        }
        if self.compress_ratio:
            meta.update(infer_state.v41_sparse_selection)
        return meta

    def _write_global_kv(self, x, infer_state: DeepseekV41InferStateInfo, layer_weight):
        ratio = self.compress_ratio
        projected = self.alloc_tensor(
            (x.shape[0], self.head_dim_ * ratio), dtype=torch.float32 if ratio == 2 else x.dtype, device=x.device
        )
        layer_weight.compressor_wkv_gate_.mm(x, out=projected)
        if ratio == 2:
            kv, score = projected.split(self.head_dim_, -1)
            latent = pool_pairs(
                kv,
                score,
                infer_state.dsv4_sparse_req_idx,
                infer_state.position_ids,
                infer_state.req_manager.req_to_c4_state[
                    infer_state.req_manager.cache_config.layer_to_c4_idx[self.layer_num_], :, 0, : 2 * self.head_dim_
                ].unflatten(-1, (2, self.head_dim_)),
                infer_state.req_manager.HOLD_REQUEST_ID,
            )
        else:
            latent = projected
        latent = layer_weight.compressor_norm_(latent, eps=self.eps_)
        positions = (infer_state.position_ids - ratio + 1).clamp_min(0)
        # Indexer K consumes the normalized latent before main KV's RoPE/FP4 quantization.
        key = layer_weight.idx_k_norm_(layer_weight.idx_wk_.mm(latent), eps=self.eps_)
        key = rotary_quantize_fp4(key, self.freqs_cis, positions)
        cache = get_indexer_k_buffer(infer_state.mem_manager, self.layer_num_)
        destindex_copy_indexer_k_dsv4(
            key,
            infer_state.mem_index,
            infer_state.position_ids,
            cache,
            infer_state.req_manager.cache_config.history_token_alignment // ratio,
            compress_ratio=ratio,
        )
        main = rotary_quantize_fp4(latent, self.freqs_cis, positions, 16)
        destindex_copy_kv_dsv41(
            main,
            infer_state.mem_index,
            infer_state.position_ids,
            get_compressed_kv_buffer(infer_state.mem_manager, self.layer_num_),
            ratio,
            infer_state.req_manager.cache_config.history_token_alignment // ratio,
        )

    def _build_index_metadata(self, x, q_lora, infer_state: DeepseekV41InferStateInfo, layer_weight):
        ratio = self.compress_ratio
        source = infer_state.req_manager.cache_config.layer_to_kv_source[self.layer_num_]
        cache = get_indexer_k_buffer(infer_state.mem_manager, source)
        query = layer_weight.idx_wq_b_.mm(q_lora).view(-1, self.network_config_["index_n_heads"], 128)
        query = rotary_quantize_fp4(query, self.freqs_cis, infer_state.position_ids)
        weights = self.alloc_tensor((x.shape[0], query[0].shape[1]), dtype=torch.float32, device=x.device)
        layer_weight.idx_weights_proj_.mm(x, out=weights)
        weights *= 128 ** -0.5 * query[0].shape[1] ** -0.5
        candidates = infer_state.v41_candidates if self.uses_candidates else None
        count = infer_state.v41_index_capacity // ratio if candidates is None else candidates.shape[1] * 8
        row_budget = max(1, (64 * 1024 ** 2) // max(4 * count, 1))
        selections, candidate_rows = [], []
        for start in range(0, x.shape[0], row_budget):
            end = min(x.shape[0], start + row_budget)
            logits = index_logits(
                (query[0][start:end], query[1][start:end]),
                weights[start:end],
                cache,
                infer_state.dsv4_sparse_req_idx[start:end],
                infer_state.position_ids[start:end],
                infer_state.req_manager.req_to_token_indexs,
                ratio,
                count,
                None if candidates is None else candidates[start:end],
            )
            k = min(count, self.index_topk)
            values, indices = logits.topk(k, -1, sorted=False)
            if candidates is not None:
                positions = (candidates[start:end, :, None] * 8 + torch.arange(8, device=x.device)).flatten(1)
                indices = positions.gather(1, indices)
            invalid = torch.iinfo(indices.dtype).max
            indices = indices.masked_fill(~values.isfinite(), invalid).sort(-1).values
            indices = indices.masked_fill(indices == invalid, -1)
            selections.append(F.pad(indices, (0, self.index_topk - k), value=-1))
            if self.is_candidate_source:
                scores = F.pad(logits, (0, -count % 8), value=-float("inf")).unflatten(-1, (-1, 8)).amax(-1)
                recent = ((infer_state.position_ids[start:end] + 1) // ratio - 1) // 8
                scores = scores.masked_fill(
                    torch.arange(scores.shape[1], device=x.device)[None, :] == recent[:, None], float("inf")
                )
                top = scores.topk(min(scores.shape[1], self.candidate_blocks), -1)
                candidate_rows.append(torch.where(top.values > -float("inf"), top.indices, -1))
        logical = torch.cat(selections, 0)
        if self.is_candidate_source:
            infer_state.v41_candidates = torch.cat(candidate_rows, 0)
        req = infer_state.dsv4_sparse_req_idx.long()
        full_positions = logical.clamp_min(0) * ratio + ratio - 1
        slots = infer_state.req_manager.req_to_token_indexs[req[:, None], full_positions].long() // ratio
        valid = (logical >= 0) & (req[:, None] != infer_state.req_manager.HOLD_REQUEST_ID)
        return {
            "extra_indices": torch.where(valid, slots, -1).to(torch.int32)[:, None, :],
            "extra_lengths": valid.sum(-1).to(torch.int32),
        }
