"""Text-only V4.1 integration using the existing V4 model and service interfaces."""

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from .infer_struct import DeepseekV41InferStateInfo
from .layer_weights.pre_and_post_layer_weight import DeepseekV41PreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
from .layer_infer.post_layer_infer import DeepseekV41PostLayerInfer
from .layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer
from .engram import EngramHasher
from lightllm.common.build_utils import repair_config
from lightllm.utils.config_utils import get_config_json


class DeepseekV41TpPartModel(DeepseekV4TpPartModel):
    pre_and_post_weight_class = DeepseekV41PreAndPostLayerWeight
    transformer_weight_class = DeepseekV41TransformerLayerWeight
    transformer_layer_infer_class = DeepseekV41TransformerLayerInfer
    post_layer_infer_class = DeepseekV41PostLayerInfer
    infer_state_class = DeepseekV41InferStateInfo

    def _init_config(self):
        # Normalize the nested checkpoint before the base model creates its flat aliases.
        self.config = get_config_json(self.weight_dir_)
        text = self.config.pop("text_config", {})
        text.pop("model_type", None)
        for key, value in text.items():
            self.config.setdefault(key, value)
        self.config.setdefault("num_hash_layers", 0)
        self.config.setdefault("expert_dtype", self.config.get("quantization_config", {}).get("expert_dtype"))
        repair_config(self.config, ["num_attention_heads", "n_head"])
        repair_config(self.config, ["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, ["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def _verify_params(self):
        super()._verify_params()
        args = self.args
        if (
            args.run_mode != "normal"
            or args.mtp_mode is not None
            or args.enable_cpu_cache
            or args.enable_dp_prompt_cache_fetch
            or args.enable_prefill_cudagraph
            or args.enable_prefill_microbatch_overlap
            or args.enable_decode_microbatch_overlap
            or self.enable_tpsp_mix_mode
        ):
            raise ValueError(
                "DeepSeek-V4.1 currently supports normal text serving with TP or DP/EP; "
                "MTP, CPU cache, prefill CUDA Graph and overlap are unsupported"
            )
        cfg = self.config
        if cfg["candidate_block_size"] != 8:
            raise ValueError("DeepSeek-V4.1 hierarchical indexer requires candidate_block_size=8")
        sources = cfg["kv_source_layer_ids"]
        index_sources = cfg["index_source_layer_ids"]
        if sources != sorted(set(sources)) or index_sources != sorted(set(index_sources)):
            raise ValueError("DeepSeek-V4.1 source layer declarations must be strictly increasing")
        if not set(sources).issubset(index_sources):
            raise ValueError("Every DeepSeek-V4.1 KV source must also generate indices")

    def _init_custom(self):
        super()._init_custom()
        self.engram_hasher = None
        if self.config["engram_layer_ids"]:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.engram_hasher = EngramHasher(
                self.config, tokenizer, self.mem_manager.size + self.mem_manager.page_size
            )
