from lightllm.models.qwen3_eagle.adapter import Qwen3EagleModelAdapter
import copy
from typing import List

import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.qwen3_eagle.layer_infer.pre_layer_infer import Qwen3EaglePreLayerInfer
from lightllm.models.qwen3_eagle.layer_infer.transformer_layer_infer import Qwen3EagleTransformerLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight


class Qwen3EagleModel(LlamaTpPartModel):
    is_mtp_draft_model = True
    model_adapter_class = Qwen3EagleModelAdapter

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")

    @classmethod
    def normalize_model_config(cls, model_config):
        config = super().normalize_model_config(model_config)
        draft_vocab_size = config.get("draft_vocab_size")
        if draft_vocab_size is not None:
            config.setdefault("target_vocab_size", config["vocab_size"])
            config["vocab_size"] = draft_vocab_size
        return config

    def _verify_params(self):
        super()._verify_params()
        assert not self.enable_tpsp_mix_mode, "Qwen3 Eagle draft model does not support TP-SP"

    def _init_custom(self):
        self._cos_cached = self.main_model._cos_cached
        self._sin_cached = self.main_model._sin_cached

    def _create_model_adapter(self, adapter_class):
        start = len(self.main_model.layers_infer) + sum(
            len(model.layers_infer) for model in self.mtp_previous_draft_models
        )
        return adapter_class(
            self.config,
            role=self.att_state_role,
            dtype=self.data_type,
            tp_world_size=self.tp_world_size_,
            layer_ids=range(start, start + self.config["num_hidden_layers"]),
            shared_embedding=self.main_model.pre_post_weight.wte_weight_,
        )

    def _create_inferstate(self, model_input: ModelInput, microbatch_index: int = 0):
        """在进入模型主体和 CUDA Graph 前统一 EAGLE3 hidden 的宽度。

        Target model 收集的多个辅助层 hidden 会沿最后一维拼接，宽度为
        ``target_layer_num * hidden_size``；后续自回归 draft step 返回的 hidden
        已经是 ``hidden_size``。Decode CUDA Graph 只按 batch size 缓存，因此
        不能让这两种 shape 和对应的 FC 分支进入同一张图。

        这里在创建 InferState 之前完成一次 target hidden 投影，使 prefill、
        decode、Graph capture 和 replay 看到的输入始终为 ``[token_num, hidden_size]``。
        使用浅副本避免覆盖调用方持有的 target ModelInput。
        """

        draft_hiddens = model_input.mtp_draft_input_hiddens
        assert draft_hiddens is not None
        assert draft_hiddens.ndim == 2
        assert draft_hiddens.shape[0] == model_input.input_ids.shape[0]

        hidden_size = int(self.config["hidden_size"])
        if draft_hiddens.shape[-1] != hidden_size:
            model_input = copy.copy(model_input)
            model_input.mtp_draft_input_hiddens = self.pre_post_weight.fc_weight_.mm(draft_hiddens)

        assert model_input.mtp_draft_input_hiddens.shape == (model_input.input_ids.shape[0], hidden_size)
        return super()._create_inferstate(model_input=model_input, microbatch_index=microbatch_index)

    # d2t stores per-token offsets: target_id = draft_id + d2t[draft_id].
    @torch.no_grad()
    def map_draft_vocab_to_main_vocab(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.pre_post_weight.d2t_weight_ is not None:
            draft_token_ids = draft_token_ids + self.pre_post_weight.d2t_weight_.weight[draft_token_ids].to(
                dtype=draft_token_ids.dtype
            )
        return draft_token_ids
