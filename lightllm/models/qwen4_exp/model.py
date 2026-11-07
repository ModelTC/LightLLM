import json
from pathlib import Path

from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.utils.torch_dtype_utils import get_torch_dtype
from .configuration import normalize_model_config
from .infer_struct import Qwen4ExpInferStateInfo
from .layer_weights.pre_and_post_layer_weight import Qwen4ExpPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from .layer_infer.pre_layer_infer import Qwen4ExpPreLayerInfer
from .layer_infer.post_layer_infer import Qwen4ExpPostLayerInfer
from .layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer


class Qwen4ExpTpPartModel(Qwen3_5MOETpPartModel):
    linear_att_mtp_match_decode = True
    pre_and_post_weight_class = Qwen4ExpPreAndPostLayerWeight
    transformer_weight_class = Qwen4ExpTransformerLayerWeight
    pre_layer_infer_class = Qwen4ExpPreLayerInfer
    post_layer_infer_class = Qwen4ExpPostLayerInfer
    transformer_layer_infer_class = Qwen4ExpTransformerLayerInfer
    infer_state_class = Qwen4ExpInferStateInfo

    def _init_config(self):
        with (Path(self.weight_dir_) / "config.json").open() as stream:
            checkpoint = json.load(stream)
        self.config, self.vision_config = normalize_model_config(checkpoint, self.finetune_config)
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    @classmethod
    def normalize_model_config(cls, model_config):
        config = normalize_model_config(model_config)[0]
        interval = config["full_attention_interval"]
        expected = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(config["num_hidden_layers"])
        ]
        actual = [
            "full_attention" if kind == "indexed_attention" else kind for kind in config.get("layer_types", expected)
        ]
        if actual != expected:
            raise ValueError("Qwen4Exp requires the configured periodic attention layout")
        config["layer_types"] = expected
        return config

    @classmethod
    def create_full_att_states(cls, model_config, args, model_logical_layer_ids, tp_rank=0, tp_world_size=1):
        from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAAttentionConfig
        from .attention_state import Qwen4ExpAttentionState

        config = cls.normalize_model_config(model_config)
        if args.llm_kv_type != "None":
            raise ValueError("Qwen4Exp QSA currently requires unquantized KV storage")
        interval = config["full_attention_interval"]
        layers = tuple((layer,) for index, layer in enumerate(model_logical_layer_ids) if (index + 1) % interval == 0)
        return [
            Qwen4ExpAttentionState(
                GQAAttentionConfig(
                    logical_layer_groups=layers,
                    global_head_num=config["num_key_value_heads"],
                    head_dim=config["head_dim"],
                    dtype=get_torch_dtype(args.data_type),
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                ),
                config["indexer_head_dim"],
                int(config["head_dim"] * config["partial_rotary_factor"]),
                args.page_size,
            )
        ]

    @classmethod
    def create_bounded_att_states(
        cls, model_config, args, model_logical_layer_ids, request_capacity=1, tp_rank=0, tp_world_size=1, device="meta"
    ):
        from .lexical_state import LexicalStateConfig, Qwen4ExpLexicalState
        from .checkpoint import ngram_eos_token_id

        config = cls.normalize_model_config(model_config)
        states = super().create_bounded_att_states(
            config, args, model_logical_layer_ids, request_capacity, tp_rank, tp_world_size, device
        )
        layers = tuple(model_logical_layer_ids[layer - 1] for layer in config.get("ple_layer_ids", []))
        if layers:
            states.append(
                Qwen4ExpLexicalState(
                    LexicalStateConfig(
                        logical_layer_ids=layers,
                        ngram_context=config["ngram_size"] - 1,
                        conv_context=(config["ple_conv_kernel_size"] - 1) * config["ngram_size"],
                        width=config["hc_count"] * config["hidden_size"],
                        eos_token_id=ngram_eos_token_id(config),
                        dtype=get_torch_dtype(args.data_type),
                        tp_rank=tp_rank,
                    ),
                    request_capacity,
                    args.mtp_step,
                    device,
                )
            )
        return states

    def prepare_attention_states(self):
        super().prepare_attention_states()
        from .lexical_state import Qwen4ExpLexicalState

        for state in self.att_state_manager.get_bounded_attention_states(self.att_state_role):
            if isinstance(state, Qwen4ExpLexicalState):
                state.bind_weights(self.trans_layers_weight)

    def _init_att_backend(self):
        self.prefill_att_backend = self.decode_att_backend = TritonAttBackend(model=self)
