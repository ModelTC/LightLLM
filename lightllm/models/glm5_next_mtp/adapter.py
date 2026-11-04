from copy import deepcopy
from lightllm.models.glm5_next_mtp.layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
from lightllm.models.glm5_next_mtp.layer_weights.pre_and_post_layer_weight import Glm5NextMTPPreAndPostLayerWeight
from lightllm.models.glm5_next.adapter import Glm5NextModelAdapter
from lightllm.models.base.adapter import grouped_moe_specs


class Glm5NextMTPModelAdapter(Glm5NextModelAdapter):
    pre_and_post_weight_class = Glm5NextMTPPreAndPostLayerWeight
    pre_layer_infer_class = Glm5NextMTPPreLayerInfer

    def __init__(self, config, *, shared_weights, **kwargs):
        config = deepcopy(config)
        if config["num_hidden_layers"] != 1 or config.get("num_nextn_predict_layers") != 1:
            raise ValueError("GLM NextN requires one native MTP block")
        config["layer_types"] = ["deepseek_sparse_attention"]
        config["mhc"] = False
        super().__init__(config, **kwargs)
        self.checkpoint_layer_id = config["mtp_checkpoint_layer_start"]
        if self.checkpoint_layer_id < 0:
            raise ValueError("GLM MTP checkpoint layer must be nonnegative")
        self.shared_weights = shared_weights

    def moe_specs(self):
        specs = grouped_moe_specs(
            self.config,
            self.dtype,
            layer_ids=(self.checkpoint_layer_id,),
            group_score_topk=2,
            limit=self.config["swiglu_limit"],
            clamp_up_add_one=False,
        )
        return {self.layer_ids[0]: specs[self.checkpoint_layer_id]} if specs else {}

    def build_weights(self, quant_config):
        pre_post = self.pre_and_post_weight_class(self.dtype, self.config, quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_, pre_post.main_norm_weight_ = self.shared_weights
        layer = self.transformer_weight_class(self.checkpoint_layer_id, self.dtype, self.config, quant_config)
        return pre_post, [layer]

    def build_recipe(self):
        pre = self.pre_layer_infer_class(self.config)
        post = self.post_layer_infer_class(self.config)
        layer = self.transformer_layer_infer_class(self.checkpoint_layer_id, self.config)
        layer.layer_num_ = self.layer_ids[0]
        layer.indexer.layer_idx = layer.layer_num_
        self.validate_layer(layer)
        return pre, post, [layer]
