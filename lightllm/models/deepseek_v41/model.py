from transformers import PreTrainedTokenizerFast

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from .engram import EngramHasher
from .infer_struct import DeepseekV41InferStateInfo
from .layer_infer.post_layer_infer import DeepseekV41PostLayerInfer
from .layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer
from .layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight


class DeepseekV41TpPartModel(DeepseekV4TpPartModel):
    transformer_weight_class = DeepseekV41TransformerLayerWeight
    transformer_layer_infer_class = DeepseekV41TransformerLayerInfer
    post_layer_infer_class = DeepseekV41PostLayerInfer
    infer_state_class = DeepseekV41InferStateInfo

    @classmethod
    def normalize_model_config(cls, model_config):
        config = super().normalize_model_config(model_config)
        config.update({key: value for key, value in model_config.items() if key != "text_config"})
        config["num_hash_layers"] = 0
        config["vision_n_layers"] = (config.get("vision_config") or {}).get("num_hidden_layers", 0)
        config.setdefault("expert_dtype", config.get("quantization_config", {}).get("expert_dtype"))
        return config

    def _verify_params(self):
        super()._verify_params()
        if (
            self.enable_tpsp_mix_mode
            or self.args.enable_prefill_microbatch_overlap
            or self.args.enable_decode_microbatch_overlap
        ):
            raise ValueError("DeepSeek-V4.1 does not support TP-SP or microbatch overlap")
        if self.args.enable_prefill_cudagraph:
            raise ValueError("DeepSeek-V4.1 requires eager prefill")
        assert self.config["candidate_block_size"] == 8, "the V4.1 indexer selects blocks of eight keys"

    def init_runtime(self):
        # History uses the same physical token slots as full attention, including the HOLD page.
        allocator = self.att_state_manager.full_att_allocator
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.weight_dir_)
        self.engram_hasher = EngramHasher(self.config, tokenizer, allocator.size + allocator.page_size)
        self.token_lookback_depth = self.engram_hasher.order - 1
        super().init_runtime()

    def forward(self, model_input):
        if model_input.token_lookback:
            self.engram_hasher.restore_history(model_input.token_lookback, self.req_manager)
        return super().forward(model_input)


class DeepseekV41VisionTpPartModel(DeepseekV41TpPartModel):
    has_vision = True
