import torch
from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_eagle.layer_infer.pre_layer_infer import Qwen3EaglePreLayerInfer
from lightllm.models.qwen3_eagle.layer_infer.transformer_layer_infer import Qwen3EagleTransformerLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight


class Qwen3EagleModel(LlamaTpPartModel):
    pre_and_post_weight_class = Qwen3EaglePreAndPostLayerWeight
    pre_layer_infer_class = Qwen3EaglePreLayerInfer

    transformer_weight_class = Qwen3EagleTransformerLayerWeight
    transformer_layer_infer_class = Qwen3EagleTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)
        return

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        return

    def _gen_special_model_input(self, token_num: int):
        return self._gen_mtp_draft_special_model_input(token_num)

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, quant_cfg=self.quant_cfg
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models
        )
        super()._init_infer_layer(start_layer_index=total_pre_layers_num)
        return

    # d2t stores per-token offsets: target_id = draft_id + d2t[draft_id].
    @torch.no_grad()
    def map_draft_vocab_to_main_vocab(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.pre_post_weight.d2t_weight_ is not None:
            draft_token_ids = draft_token_ids + self.pre_post_weight.d2t_weight_.weight[draft_token_ids].to(
                dtype=draft_token_ids.dtype
            )
        return draft_token_ids
