import torch
from typing import List

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen3_eagle.layer_infer.transformer_layer_infer import Qwen3EagleTransformerLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight


class Qwen3EagleModel(LlamaTpPartModel):
    pre_and_post_weight_class = Qwen3EaglePreAndPostLayerWeight

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

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached
        return

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager
        return

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.tp_world_size_,
            head_dim=self.config["head_dim"],
            layer_num=1,
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        mtp_index = len(self.mtp_previous_draft_models)
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
            for i in range(mtp_index, mtp_index + self.config["n_layer"])
        ]
        
        if self.pre_post_weight.wte_weight_ is None:
            self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
            
        if self.pre_post_weight.lm_head_weight_ is None:
            self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
            
        if self.pre_post_weight.final_norm_weight_ is None:
            self.pre_post_weight.final_norm_weight_ = self.main_model.pre_post_weight.final_norm_weight_
        return

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        # total_pre_layers_num = len(self.main_model.layers_infer)
        # total_pre_layers_num += sum(
        #     [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        # )
        total_pre_layers_num = 0
        super()._init_infer_layer(start_layer_index=total_pre_layers_num)
        return
    
    # For Eagle3 Model, we need to expose a function to do the mapping of draft vocab and main vocab
    # Using d2t and t2d weight to do the mapping
    @torch.no_grad()
    def map_draft_vocab_to_main_vocab(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.pre_post_weight.d2t_weight_ is not None:
            draft_token_ids = draft_token_ids + self.pre_post_weight.d2t_weight_.weight[draft_token_ids]
        return draft_token_ids

    # Override forward function, for the pre-compute the mtp_draft_input_hiddens
    @torch.no_grad()
    def forward(self, model_input: ModelInput):
        model_input.to_cuda()
        assert model_input.mem_indexes.is_cuda
        
        if model_input.mtp_draft_input_hiddens.shape[-1] != self.config["hidden_size"]:
            model_input.mtp_draft_input_hiddens = self.trans_layers_weight[0].fc_weight_.mm(model_input.mtp_draft_input_hiddens)

        if model_input.is_prefill:
            return self._prefill(model_input)
        else:
            return self._decode(model_input)
