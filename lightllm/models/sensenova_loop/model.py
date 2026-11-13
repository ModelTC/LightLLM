"""SenseNova Loop: shared checkpoint weights and independent execution caches."""

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from .config import get_loop_layer_indices, get_loop_layer_types
from .layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
from .layer_weights.transformer_layer_weight import SenseNovaLoopTransformerLayerWeight
from .dsa import validate_dsa_config
from .infer_struct import DsaLoopInferStateInfo


class SenseNovaLoopTpPartModel(Qwen3_5TpPartModel):
    """Share physical weights, with separate KV/conv/SSM slots per execution layer."""

    transformer_weight_class = SenseNovaLoopTransformerLayerWeight
    transformer_layer_infer_class = SenseNovaLoopTransformerLayerInfer
    infer_state_class = DsaLoopInferStateInfo

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        config = super().normalize_model_config(model_config)
        if "loop_physical_indices" not in config:
            settings = model_config["loop_settings"]
            indices = get_loop_layer_indices(settings, config["num_hidden_layers"])
            config.update(
                layer_types=get_loop_layer_types(settings, config),
                n_layer=len(indices),
                num_hidden_layers=len(indices),
                loop_physical_indices=indices,
            )
        if model_config.get("mhc") is not None:
            config["mhc"] = model_config["mhc"]
        dsa = validate_dsa_config(model_config)
        if dsa is not None:
            config["dsa"] = dsa
            config["dsa_export"] = dict(model_config["dsa_export"])
        return config

    def _init_config(self):
        super()._init_config()
        self.dsa = self.config.get("dsa")
        self.loop_layer_indices = self.config["loop_physical_indices"]
        self.num_physical_layers = max(self.loop_layer_indices) + 1

    def _init_weights(self, start_layer_index=0):
        # Allocate and preprocess each checkpoint layer once, then alias weights.
        self.pre_post_weight = self.pre_and_post_weight_class(self.data_type, network_config=self.config)
        self.physical_layer_weights = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(start_layer_index, start_layer_index + self.num_physical_layers)
        ]
        self.trans_layers_weight = [self.physical_layer_weights[i] for i in self.loop_layer_indices]

    def _init_infer_layer(self, start_layer_index=0):
        super()._init_infer_layer(start_layer_index)
        # Bind split projections once for each visit to a shared physical layer.
        for layer, weight in zip(self.layers_infer, self.trans_layers_weight):
            if getattr(weight, "linear_gate_proj", None) is not None:
                layer._linear_in_proj = layer._linear_in_proj_with_gates

    def load_weights(self, weight_dict):
        execution_weights = self.trans_layers_weight
        self.trans_layers_weight = self.physical_layer_weights
        try:
            super().load_weights(weight_dict)
        finally:
            self.trans_layers_weight = execution_weights

    @classmethod
    def create_full_att_states(cls, model_config, args, model_logical_layer_ids, tp_rank=0, tp_world_size=1):
        states = super().create_full_att_states(model_config, args, model_logical_layer_ids, tp_rank, tp_world_size)
        config = cls.normalize_model_config(model_config)
        dsa = config.get("dsa")
        if dsa is None or not states:
            return states
        import torch
        from lightllm.common.attention_state.full_att import MLAConfig
        from lightllm.utils.torch_dtype_utils import get_torch_dtype
        from .dsa_state import DsaIndexerState

        if args.llm_kv_type != "None" or get_torch_dtype(args.data_type) != torch.bfloat16:
            raise ValueError("SenseNova DSA v1 requires BF16 weights and unquantized KV")
        states.append(
            DsaIndexerState(
                layout_config=MLAConfig(
                    logical_layer_groups=states[0].config.logical_layer_groups,
                    latent_dim=dsa["head_dim"],
                    latent_dtype=torch.bfloat16,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world_size,
                ),
                page_size=args.page_size,
                device="meta",
            )
        )
        return states

    def _init_to_get_yarn_rotary(self):
        if not self.dsa:
            return super()._init_to_get_yarn_rotary()
        from .dsa import yarn_cos_sin

        self._cos_cached, self._sin_cached = yarn_cos_sin(self.config, self.max_seq_length, "cuda")

    def _init_cudagraph(self):
        if not self.dsa:
            return super()._init_cudagraph()
        from .dsa_graph import DsaCudaGraph

        if self.args.mtp_mode is not None and self.graph_max_len_in_batch >= self.args.max_req_total_len:
            self.graph_max_len_in_batch = max(self.graph_max_len_in_batch, self.max_seq_length)
        per_request = self.mtp_manager.get_decode_tokens_per_request(self.is_mtp_draft_model)
        grow = self.mtp_manager.get_decode_batch_alignment(self.is_mtp_draft_model)
        self.graph = (
            None
            if self.disable_cudagraph
            else DsaCudaGraph(
                batch_step_size_before_split=grow,
                split_batch_size=self.args.graph_split_batch_size * per_request,
                batch_step_size_after_split=self.args.graph_grow_step_size * grow,
                max_batch_size=self.graph_max_batch_size,
                max_len_in_batch=self.graph_max_len_in_batch,
                tp_world_size=self.tp_world_size_,
                capture_infer_cost=self.args.mtp_dynamic_verify,
            )
        )
        if self.graph is not None:
            if self.args.enable_decode_microbatch_overlap:
                self.graph.warmup_overlap(self)
            else:
                self.graph.warmup(self)
