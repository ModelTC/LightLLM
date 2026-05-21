import os
import json
from typing import List
from lightllm.models.gemma4.model import Gemma4TpPartModel
from lightllm.models.gemma4_mtp.layer_weights.pre_and_post_layer_weight import Gemma4MTPPreAndPostLayerWeight
from lightllm.models.gemma4_mtp.layer_weights.transformer_layer_weight import Gemma4MTPTransformerLayerWeight
from lightllm.models.gemma4_mtp.layer_infer.pre_layer_infer import Gemma4MTPPreLayerInfer
from lightllm.models.gemma4_mtp.layer_infer.transformer_layer_infer import Gemma4MTPTransformerLayerInfer
from lightllm.models.gemma4_mtp.layer_infer.post_layer_infer import Gemma4MTPPostLayerInfer
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Gemma4MTPModel(Gemma4TpPartModel):
    """
    Gemma-4 assistant drafter (frozen-KV MTP). Subclasses the Gemma-4 target model
    and reuses its decoder block / RoPE tables / attention backends, but:
      * shares the target's mem_manager and req_manager - the drafter allocates no
        KV of its own (frozen-KV: it reads the target's committed cache),
      * builds only num_hidden_layers (4) draft layers, each forced into the
        KV-shared Q-only attention path pointed at a target-model layer,
      * fuses the target's recurrent hidden state with the token embedding via
        pre_projection / post_projection (see Gemma4MTPPreLayerInfer).

    Instantiated directly by ModeBackend.init_mtp_draft_model - not registered via
    @ModelRegistry and not imported in lightllm/models/__init__.py.
    """

    pre_and_post_weight_class = Gemma4MTPPreAndPostLayerWeight
    transformer_weight_class = Gemma4MTPTransformerLayerWeight

    pre_layer_infer_class = Gemma4MTPPreLayerInfer
    transformer_layer_infer_class = Gemma4MTPTransformerLayerInfer
    post_layer_infer_class = Gemma4MTPPostLayerInfer

    def __init__(self, kvargs):
        self._pre_init(kvargs)
        super().__init__(kvargs)

    def _pre_init(self, kvargs):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as f:
            outer_cfg = json.load(f)
        super()._init_config()
        # backbone_hidden_size lives on the outer assistant config, above text_config.
        self.config["backbone_hidden_size"] = outer_cfg["backbone_hidden_size"]
        # E-series centroid sparse-logits head (gemma-4-E4B-it-assistant etc.):
        # forward outer-config fields used by Gemma4MTPPreAndPostLayerWeight /
        # Gemma4MTPPostLayerInfer to mask draft logits to the top-K centroids'
        # vocab slice.
        self.config["use_ordered_embeddings"] = bool(outer_cfg.get("use_ordered_embeddings"))
        if self.config["use_ordered_embeddings"]:
            self.config["num_centroids"] = outer_cfg["num_centroids"]
            self.config["centroid_intermediate_top_k"] = outer_cfg["centroid_intermediate_top_k"]
        # The assistant config marks every layer KV-shared - this denotes
        # cross-model sharing with the target, not the intra-model sharing the
        # inherited layer infer expects. Force it to 0 so
        # Gemma4TransformerLayerInfer.__init__ leaves is_kv_shared_ False;
        # Gemma4MTPTransformerLayerInfer then sets it True with the correct
        # target-model layer index.
        self.config["num_kv_shared_layers"] = 0

    def _init_custom(self):
        # Reuse the target's RoPE tables (built for the target's max_seq_len, which
        # covers every position the drafter ever queries). Skip the deepep group
        # setup - the assistant trunk is always dense.
        self._cos_cached_sliding = self.main_model._cos_cached_sliding
        self._sin_cached_sliding = self.main_model._sin_cached_sliding
        self._cos_cached_full = self.main_model._cos_cached_full
        self._sin_cached_full = self.main_model._sin_cached_full

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager

    def _init_mem_manager(self):
        # Frozen-KV: the drafter never writes KV, it reads the target's cache.
        self.mem_manager = self.main_model.mem_manager

    def _init_weights(self):
        super()._init_weights()
        # The input token embedding is the target's (backbone width); the
        # assistant's own model.embed_tokens serves only as the tied lm_head.
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        # post_projection lives in pre_post_weight; the pre_infer needs it in its
        # _tpsp_allgather override to lift the trunk output to backbone width.
        self.pre_infer._post_projection_weight_ = self.pre_post_weight.post_projection_weight_

        # Map each assistant layer to the target model's layer whose KV cache it
        # reads: the target's last non-KV-shared layer of the same attention type.
        target_cfg = self.main_model.config
        target_layer_types = target_cfg["layer_types"]
        target_kv_shared = target_cfg.get("num_kv_shared_layers") or 0
        target_cutoff = len(target_layer_types) - target_kv_shared
        last_of_type = {}
        for j in range(target_cutoff):
            last_of_type[target_layer_types[j]] = j

        draft_layer_types = self.config["layer_types"]
        self.layers_infer = []
        for i in range(self.config["num_hidden_layers"]):
            layer_type = draft_layer_types[i]
            self.layers_infer.append(
                self.transformer_layer_infer_class(
                    i, network_config=self.config, kv_share_target_layer=last_of_type[layer_type]
                )
            )

    def autotune_layers(self):
        return self.config["num_hidden_layers"]
