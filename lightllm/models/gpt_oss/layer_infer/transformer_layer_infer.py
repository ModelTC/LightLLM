import torch
from lightllm.common.basemodel.attention_backend_manager import LayerAttentionConfig
from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class GptOssTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.hidden_size = self.network_config_["hidden_size"]
        self.alpha = 1.702
        self.limit = 7.0
        self.top_k = network_config["num_experts_per_tok"]
        self.sliding_window = network_config["sliding_window"]
        self.head_dim_ = network_config["head_dim"]

    def _bind_norm(self):
        self._att_norm = self._att_norm
        self._ffn_norm = self._ffn_norm
        return

    def _att_norm(self, input, infer_state, layer_weight: GptOssTransformerLayerWeight) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_)
        return out

    def _ffn_norm(self, input, infer_state, layer_weight: GptOssTransformerLayerWeight) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_)
        return out

    def _gpt_oss_rmsnorm(self, hidden_states, weight, eps=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)  # main diff with Llama

    def _ffn(self, input, infer_state, layer_weight: GptOssTransformerLayerWeight) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        hidden_states = self._tpsp_allgather(input=hidden_states, infer_state=infer_state)
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        hidden_states = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=True,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
            infer_state=infer_state,
        )
        hidden_states = hidden_states.view(num_tokens, hidden_dim)
        return self._tpsp_reduce(input=hidden_states, infer_state=infer_state)

    def attention_config(self, model) -> LayerAttentionConfig:
        is_sliding = self.network_config_["layer_types"][self.layer_num_] == "sliding_attention"
        window = (self.sliding_window - 1, 0) if is_sliding else (-1, -1)
        config = self._gqa_config(model, has_sinks=True, sliding_window=window)
        return LayerAttentionConfig(prefill=config, decode=config)

    def _gqa_sinks(self, layer_weight: GptOssTransformerLayerWeight):
        return layer_weight.attn_sinks.weight
