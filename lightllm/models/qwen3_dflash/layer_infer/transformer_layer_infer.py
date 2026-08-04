import torch

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


class Qwen3DFlashTransformerLayerInfer(LlamaTransformerLayerInfer):
    """DFlash layer inference.

    The model path is built from two explicit layer primitives:
    - commit accepted target hidden rows into draft KV
    - run one non-causal draft block over prefix KV + scratch KV
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.block_size_ = int(network_config["block_size"])
        return

    def context_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        layer_weight: Qwen3DFlashTransformerLayerWeight,
    ) -> torch.Tensor:
        token_num, _ = input_embdings.shape
        kv = layer_weight.kv_proj.mm(input_embdings, use_custom_tensor_mananger=False)
        kv = kv.view(token_num, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        k = kv[:, : self.tp_k_head_num_, :]
        v = kv[:, self.tp_k_head_num_ :, :]
        k = layer_weight.k_norm_weight_(
            input=k.reshape(-1, self.head_dim_),
            eps=self.eps_,
            alloc_func=torch.empty,
        ).view(token_num, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(
            k,
            None,
            infer_state.position_cos,
            infer_state.position_sin,
        )
        cache_kv = torch.cat([k, v], dim=1)
        self._post_cache_kv(cache_kv.contiguous(), infer_state, layer_weight)
        return input_embdings

    def token_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        layer_weight: Qwen3DFlashTransformerLayerWeight,
    ) -> torch.Tensor:
        hidden_states = input_embdings.view(-1, self.block_size_, self.embed_dim_)
        residual = hidden_states
        q, cache_kv = self._get_qkv(hidden_states, infer_state, layer_weight)
        batch_size, block_size, hidden_size = hidden_states.shape
        self._post_cache_kv(cache_kv.contiguous(), infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        o = self._get_o(o, infer_state=infer_state, layer_weight=layer_weight)
        hidden_states = residual + o.view(-1, block_size, hidden_size)

        residual = hidden_states
        ffn_input = self._ffn_norm(
            hidden_states.reshape(-1, hidden_size),
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
        ffn_out = self._ffn(ffn_input, infer_state=infer_state, layer_weight=layer_weight)
        hidden_states = residual + ffn_out.view(-1, block_size, hidden_size)
        return hidden_states.view(-1, self.embed_dim_)

    def _get_qkv(self, input, infer_state: Qwen3DFlashInferStateInfo, layer_weight: Qwen3DFlashTransformerLayerWeight):
        hidden_states = input.view(-1, self.block_size_, self.embed_dim_)
        batch_size, block_size, hidden_size = hidden_states.shape
        normed = self._att_norm(
            hidden_states.reshape(-1, hidden_size),
            infer_state=infer_state,
            layer_weight=layer_weight,
        ).view(batch_size, block_size, hidden_size)

        q = layer_weight.q_proj.mm(normed.reshape(-1, hidden_size), use_custom_tensor_mananger=False)
        kv = layer_weight.kv_proj.mm(normed.reshape(-1, hidden_size), use_custom_tensor_mananger=False)

        q = q.view(batch_size, block_size, self.tp_q_head_num_, self.head_dim_)
        kv = kv.view(batch_size, block_size, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        k = kv[:, :, : self.tp_k_head_num_, :]
        v = kv[:, :, self.tp_k_head_num_ :, :]

        q = layer_weight.q_norm_weight_(
            input=q.reshape(-1, self.head_dim_),
            eps=self.eps_,
            alloc_func=torch.empty,
        ).view(batch_size, block_size, self.tp_q_head_num_, self.head_dim_)
        k = layer_weight.k_norm_weight_(
            input=k.reshape(-1, self.head_dim_),
            eps=self.eps_,
            alloc_func=torch.empty,
        ).view(batch_size, block_size, self.tp_k_head_num_, self.head_dim_)

        rotary_emb_fwd(
            q.reshape(-1, self.tp_q_head_num_, self.head_dim_),
            k.reshape(-1, self.tp_k_head_num_, self.head_dim_),
            infer_state.position_cos,
            infer_state.position_sin,
        )
        cache_kv = torch.cat([k, v], dim=2).reshape(
            batch_size * block_size,
            self.tp_k_head_num_ + self.tp_v_head_num_,
            self.head_dim_,
        )
        return q, cache_kv
