"""V4.1 integration using the existing V4 model and service interfaces."""

import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel, DeepseekV4LayerHiddenCollector
from lightllm.common.basemodel.hidden_collector import LayerHiddenCollector
from lightllm.common.basemodel.triton_kernel.mhc import hc_contract
from .infer_struct import DeepseekV41InferStateInfo
from .layer_weights.pre_and_post_layer_weight import DeepseekV41PreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
from .layer_infer.post_layer_infer import DeepseekV41PostLayerInfer
from .layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer
from .engram import EngramHasher
from lightllm.common.build_utils import repair_config
from lightllm.utils.config_utils import get_config_json


class DeepseekV41LayerHiddenCollector(DeepseekV4LayerHiddenCollector):
    """V4.1 DSpark consumes the residual streams entering each selected layer."""

    def _load_layer_ids(self):
        layer_ids = super()._load_layer_ids()
        assert min(layer_ids) > 0, "V4.1 DSpark target layers must follow a transformer layer"
        return frozenset(layer - 1 for layer in layer_ids)

    def add(self, layer_index, hidden):
        if layer_index not in self.layer_ids:
            return
        # A layer's completed streams are the next layer's input; ignore the shifted pre-mix.
        streams = hidden[0] if isinstance(hidden, tuple) else hidden
        LayerHiddenCollector.add(self, layer_index, hc_contract(streams, self.model.config["hc_mult"]))

    def finish_output(self, infer_state):
        decoder = infer_state.decoder_state
        if decoder is not None:
            decoder_start = max(self.model.config["kv_source_layer_ids"])
            # Configurations may select an encoder layer as well as decoder layers.
            self.layer_hiddens = [
                hidden[infer_state.decoder_token_indices] if layer < decoder_start else hidden
                for layer, hidden in zip(sorted(self.layer_ids), self.layer_hiddens)
            ]
        output = super().finish_output(decoder if decoder is not None else infer_state)
        if decoder is not None:
            output.spec_hidden_prefill_ready_cache_len = decoder.b_ready_cache_len
        return output


class DeepseekV41TpPartModel(DeepseekV4TpPartModel):
    pre_and_post_weight_class = DeepseekV41PreAndPostLayerWeight
    transformer_weight_class = DeepseekV41TransformerLayerWeight
    transformer_layer_infer_class = DeepseekV41TransformerLayerInfer
    post_layer_infer_class = DeepseekV41PostLayerInfer
    infer_state_class = DeepseekV41InferStateInfo

    @property
    def prefill_replay_window(self):
        return 0 if self.return_all_prompt_logics or self.is_mtp_draft_model else self.config["sliding_window"]

    def forward(self, model_input):
        if not self.is_mtp_draft_model and model_input.token_lookback is not None:
            self.engram_hasher.restore_history(model_input.token_lookback, self.req_manager)
        return super().forward(model_input)

    def _token_forward(self, infer_state):
        if not self.is_mtp_draft_model and self.config["engram_layer_ids"]:
            # This must run inside the decode graph, rather than in state setup:
            # a replay must gather fresh embeddings from the staged hash inputs.
            first = self.config["engram_layer_ids"][0]
            infer_state.prefetch_engram(self.trans_layers_weight[first], 0)
        return super()._token_forward(infer_state)

    def _init_config(self):
        config = get_config_json(self.weight_dir_)
        self.config = {**config["text_config"], **config}
        self.config.setdefault("num_hash_layers", 0)
        self.config.setdefault("expert_dtype", config.get("quantization_config", {}).get("expert_dtype"))
        self.config.setdefault("vision_n_layers", (config.get("vision_config") or {}).get("num_hidden_layers", 0))
        repair_config(self.config, ["num_attention_heads", "n_head"])
        repair_config(self.config, ["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, ["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

    def _verify_params(self):
        super()._verify_params()
        args = self.args
        if (
            args.run_mode not in ("normal", "prefill", "decode")
            or args.mtp_mode not in (None, "dspark")
            or (args.enable_cpu_cache and args.run_mode == "decode")
            or args.enable_prefill_cudagraph
            or args.enable_prefill_microbatch_overlap
            or args.enable_decode_microbatch_overlap
            or self.enable_tpsp_mix_mode
        ):
            raise ValueError(
                "DeepSeek-V4.1 supports normal and P/D serving with TP or DP/EP; "
                "only DSpark MTP is supported; CPU cache on Decode nodes, "
                "prefill CUDA Graph and overlap are unsupported"
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

    def _init_hidden_collector(self):
        if not self.is_mtp_draft_model and self.mtp_manager.args.mtp_mode == "dspark":
            self.hidden_collector_prototype = DeepseekV41LayerHiddenCollector(self)
            # Collect completed streams once, without repeating post in the next layer.
            for layer in self.hidden_collector_prototype.layer_ids:
                self.layers_infer[layer].defer_hc_post = False
        else:
            super()._init_hidden_collector()

    def _init_custom(self):
        super()._init_custom()
        self.engram_hasher = None
        self.engram_stream = None
        if self.config["engram_layer_ids"]:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.weight_dir_, trust_remote_code=True)
            self.engram_hasher = EngramHasher(
                self.config, tokenizer, self.mem_manager.size + self.mem_manager.page_size
            )
            self.token_lookback_depth = self.engram_hasher.order - 1
            self.engram_stream = torch.cuda.Stream()
            # Once one Engram is consumed, start the next lookup rather than
            # waiting until the layer immediately preceding its consumer.
            layers = self.config["engram_layer_ids"]
            for index in range(1, len(layers)):
                self.layers_infer[layers[index - 1]].next_engram = (index, self.trans_layers_weight[layers[index]])


class DeepseekV41VisionTpPartModel(DeepseekV41TpPartModel):
    has_vision = True
