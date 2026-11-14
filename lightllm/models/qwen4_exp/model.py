import json
from pathlib import Path

import torch

from lightllm.common.basemodel.cuda_graph import CudaGraph
from .layer_infer.replicated_decoder import (
    ReplicatedDecoderLayer,
    ReplicatedDecodeGraph,
    supports_fixed_decode,
    supports_replicated_decode,
)

from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
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
    _replicated_decode = False
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
        if args.llm_kv_type not in ("None", "fp8kv_spt"):
            raise ValueError("Qwen4Exp QSA supports BF16 or unit-scale FP8 KV storage")
        if args.llm_kv_type == "fp8kv_spt" and args.kv_quant_calibration_config_path:
            raise ValueError("Qwen4Exp FP8 QSA uses unit scales and does not support KV calibration files")
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
                fp8=args.llm_kv_type == "fp8kv_spt",
                compress_ratio=config["indexer_compress_ratio"] if args.page_size > 1 else 1,
            )
        ]

    @classmethod
    def create_bounded_att_states(
        cls, model_config, args, model_logical_layer_ids, request_capacity=1, tp_rank=0, tp_world_size=1, device="meta"
    ):
        from .lexical_state import LexicalStateConfig, Qwen4ExpLexicalState
        from .checkpoint import ngram_eos_token_id

        config = cls.normalize_model_config(model_config)
        states = Qwen3_5MOETpPartModel.create_bounded_att_states.__func__(
            cls, config, args, model_logical_layer_ids, request_capacity, tp_rank, tp_world_size, device
        )
        if args.page_size > 1:
            from .qsa_compressor import QSACompressorState

            ratio = config["indexer_compress_ratio"]
            if args.page_size % ratio or args.linear_att_hash_page_size % ratio:
                raise ValueError("QSA cache pages must align to the indexer compression ratio")
            interval = config["full_attention_interval"]
            qsa_layers = tuple(
                layer for index, layer in enumerate(model_logical_layer_ids) if (index + 1) % interval == 0
            )
            states.append(
                QSACompressorState(
                    qsa_layers,
                    config["indexer_head_dim"],
                    int(config["head_dim"] * config["partial_rotary_factor"]),
                    ratio,
                    get_torch_dtype(args.data_type),
                    request_capacity,
                    args.mtp_step,
                    tp_rank,
                    tp_world_size,
                    device,
                )
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

        state = getattr(self.att_state_manager, f"{self.att_state_role}_qwen4_lexical_state", None)
        if isinstance(state, Qwen4ExpLexicalState):
            state.bind_weights(self.trans_layers_weight)

    def _init_att_backend(self):
        self.prefill_att_backend = self.decode_att_backend = TritonAttBackend(model=self)
        self.fixed_decode_config = supports_fixed_decode(self)

        if self.fixed_decode_config and not self.is_mtp_draft_model:
            for layer in self.trans_layers_weight:
                if not layer.is_linear_attention_layer and not hasattr(layer, "qsa_projection"):
                    layer.qsa_projection = torch.cat(
                        (layer.qkvo_gate_proj.mm_param.weight, layer.indexer.component.projection)
                    )

    def _init_att_backend1(self):
        states = self.att_state_manager.get_bounded_attention_states(self.att_state_role)
        if not any(state.get_state_type().startswith("gdn_") for state in states):
            self.prefill_att_backend1 = self.decode_att_backend1 = None
            return
        Qwen3_5MOETpPartModel._init_att_backend1(self)
        if self.fixed_decode_config and not self.is_mtp_draft_model:
            self.decode_att_backend1.replay_run_config = {"BV": 32, "num_warps": 4, "num_stages": 2}

    def _init_hidden_collector(self):
        super()._init_hidden_collector()
        self._init_replicated_decode()

    def _init_replicated_decode(self):
        self._replicated_decode = supports_replicated_decode(self)
        self.decode_graph_class = ReplicatedDecodeGraph if self._replicated_decode else CudaGraph
        self._replicated_layers = {}

    def _align_decode_batch_size(self, batch_size):
        if self._replicated_decode and 0 < batch_size <= 64:
            return (batch_size + 3) // 4 * 4
        return super()._align_decode_batch_size(batch_size)

    def _token_forward(self, infer_state):
        rows = infer_state.input_ids.numel()
        if not self._replicated_decode or not 0 < rows <= 64:
            return super()._token_forward(infer_state)
        if rows not in self._replicated_layers:
            self._replicated_layers[rows] = [
                ReplicatedDecoderLayer(layer.layer_num_, self.config) for layer in self.layers_infer
            ]
        hidden = self.pre_infer.token_forward(infer_state.input_ids, infer_state, self.pre_post_weight)
        for layer, weight in zip(self._replicated_layers[rows], self.trans_layers_weight):
            hidden = layer.token_forward(hidden, infer_state, weight)
        post = self.post_infer.token_forward(hidden, infer_state, self.pre_post_weight)
        infer_state.hidden_collector.add_final_hidden(hidden)
        output = self._create_model_output(post, infer_state)
        if infer_state.is_cuda_graph_capture:
            output.to_no_ref_tensor()
        return output
