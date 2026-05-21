import torch
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    ParameterWeight,
    RMSNormWeight,
    ROWMMWeight,
)


class Gemma4MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    """
    Pre/post weights for the Gemma-4 assistant (frozen-KV MTP drafter).

    Layout differs from a normal model:
      * pre_projection : Linear(2 * backbone_hidden -> draft_hidden) - fuses the
        target token embedding with the recurrent hidden state.
      * post_projection: Linear(draft_hidden -> backbone_hidden) - maps the draft
        trunk output back to backbone width for the next recurrent step.
      * model.norm     : final RMSNorm in draft_hidden width.
      * lm_head        : tied to the assistant's own model.embed_tokens.weight
        (draft_hidden width); this is NOT the input embedding.
      * wte_weight_    : the *target's* input embedding (backbone_hidden width),
        aliased from the main model in Gemma4MTPModel._init_weights - never loaded
        here.
    """

    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        draft_hidden = network_config["hidden_size"]
        backbone_hidden = network_config["backbone_hidden_size"]
        vocab_size = network_config["vocab_size"]

        self.pre_projection_weight_ = ROWMMWeight(
            in_dim=backbone_hidden * 2,
            out_dims=[draft_hidden],
            weight_names="pre_projection.weight",
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        self.post_projection_weight_ = ROWMMWeight(
            in_dim=draft_hidden,
            out_dims=[backbone_hidden],
            weight_names="post_projection.weight",
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=draft_hidden,
            weight_name="model.norm.weight",
            data_type=self.data_type_,
        )
        # The assistant ships model.embed_tokens.weight in draft_hidden width;
        # with tie_word_embeddings it serves only as the tied lm_head matrix
        # (the input embedding comes from the target model, see wte_weight_).
        self._mtp_lm_head_embed_ = EmbeddingWeight(
            dim=draft_hidden,
            vocab_size=vocab_size,
            weight_name="model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=draft_hidden,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
            embedding_weight=self._mtp_lm_head_embed_,
        )
        # The input token embedding is the *target's* (backbone_hidden width);
        # aliased from the main model in Gemma4MTPModel._init_weights.
        self.wte_weight_: EmbeddingWeight = None

        # E-series centroid sparse-logits head (only the E* assistants ship these).
        # token_ordering[v] gives the centroid id of vocab token v; at decode time
        # Gemma4MTPPostLayerInfer masks logits to the per-query top-K centroids'
        # vocab slice.
        if network_config.get("use_ordered_embeddings"):
            num_centroids = network_config["num_centroids"]
            self.centroids_weight_ = ROWMMWeight(
                in_dim=draft_hidden,
                out_dims=[num_centroids],
                weight_names="masked_embedding.centroids.weight",
                data_type=self.data_type_,
                tp_rank=0,
                tp_world_size=1,
            )
            self.token_ordering_ = ParameterWeight(
                weight_name="masked_embedding.token_ordering",
                data_type=torch.int64,
                weight_shape=(vocab_size,),
            )
        return
