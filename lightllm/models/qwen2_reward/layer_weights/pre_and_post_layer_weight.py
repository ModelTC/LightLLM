from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, RMSNormWeight, ROWMMWeight


class Qwen2RewardPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.wte_weight_ = EmbeddingWeight(
            dim=network_config["hidden_size"],
            vocab_size=network_config["vocab_size"],
            weight_name="model.embed_tokens.weight",
            data_type=data_type,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=network_config["hidden_size"], weight_name="model.norm.weight", data_type=data_type
        )
        self.score_up_weight_ = self.score_down_weight_ = None
        self._score_weights = {}

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        # Checkpoints specify the score head dimensions; its fields may span shards.
        with self.lock:
            if self.score_up_weight_ is not None:
                return
            for name in ("score.0.weight", "score.0.bias", "score.2.weight", "score.2.bias"):
                if name in weights:
                    self._score_weights[name] = weights[name]
            if len(self._score_weights) != 4:
                return
            up, down = (self._score_weights[f"score.{i}.weight"] for i in (0, 2))
            if (
                up.ndim != 2
                or down.ndim != 2
                or up.shape[1] != self.network_config_["hidden_size"]
                or down.shape[1] != up.shape[0]
                or min(*up.shape, *down.shape) <= 0
                or self._score_weights["score.0.bias"].shape != (up.shape[0],)
                or self._score_weights["score.2.bias"].shape != (down.shape[0],)
            ):
                raise ValueError("reward score head has incompatible projection or bias shapes")
            heads = []
            for index, tensor in ((0, up), (2, down)):
                head = ROWMMWeight(
                    in_dim=tensor.shape[1],
                    out_dims=[tensor.shape[0]],
                    weight_names=f"score.{index}.weight",
                    bias_names=f"score.{index}.bias",
                    data_type=self.data_type_,
                    tp_rank=0,
                    tp_world_size=1,
                )
                head.load_hf_weights(self._score_weights)
                heads.append(head)
            self.score_up_weight_, self.score_down_weight_ = heads
            self._score_weights.clear()

    def verify_load(self):
        if self.score_up_weight_ is None or self.score_down_weight_ is None:
            raise ValueError("reward checkpoint is missing score head weights or biases")
        super().verify_load()
