import gc
import os
from typing import List

import torch
from safetensors import safe_open
from tqdm import tqdm

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4_mtp.layer_infer.pre_layer_infer import DeepseekV4MTPPreLayerInfer
from lightllm.models.deepseek_v4_mtp.layer_infer.transformer_layer_infer import (
    DeepseekV4MTPTransformerLayerInfer,
)
from lightllm.models.deepseek_v4_mtp.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4MTPPreAndPostLayerWeight,
)
from lightllm.models.deepseek_v4_mtp.layer_weights.transformer_layer_weight import (
    DeepseekV4MTPTransformerLayerWeight,
)
import lightllm.utils.petrel_helper as utils
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class DeepseekV4MTPModel(DeepseekV4TpPartModel):
    is_mtp_draft_model = True

    pre_and_post_weight_class = DeepseekV4MTPPreAndPostLayerWeight
    pre_layer_infer_class = DeepseekV4MTPPreLayerInfer
    transformer_weight_class = DeepseekV4MTPTransformerLayerWeight
    transformer_layer_infer_class = DeepseekV4MTPTransformerLayerInfer

    def __init__(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop("mtp_previous_draft_models")
        super().__init__(kvargs)
        return

    @classmethod
    def normalize_model_config(cls, model_config: dict[str, object]) -> dict[str, object]:
        config = super().normalize_model_config(model_config)
        # 权重层号用于量化配置及权重构造，与各 draft 实例的执行逻辑层号分别记录。
        config.setdefault("draft_weight_layer_id", config["num_hidden_layers"])
        config.update(num_hidden_layers=1, n_layer=1)
        return config

    def _init_custom(self):
        self._freqs_cis_sliding = self.main_model._freqs_cis_sliding
        self._freqs_cis_compress = self.main_model._freqs_cis_compress
        self._cos_cached_sliding = self.main_model._cos_cached_sliding
        self._sin_cached_sliding = self.main_model._sin_cached_sliding
        self._cos_cached_compress = self.main_model._cos_cached_compress
        self._sin_cached_compress = self.main_model._sin_cached_compress
        self.dsv4_workspace = self.main_model.dsv4_workspace
        for layer in self.layers_infer:
            layer.freqs_cis = self._freqs_cis_compress if layer.compress_ratio else self._freqs_cis_sliding
            layer.cos_compress_table = self._cos_cached_compress
            layer.sin_compress_table = self._sin_cached_compress
        return

    def _init_weights(self, start_layer_index=None):
        draft_weight_layer_id = self.config["draft_weight_layer_id"]
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, network_config=self.config, quant_cfg=self.quant_cfg
        )
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = self.main_model.pre_post_weight.lm_head_weight_
        self.trans_layers_weight = [
            self.transformer_weight_class(
                draft_weight_layer_id,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
        ]
        return

    def _init_infer_layer(self, start_layer_index=None):
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        total_pre_layers_num = len(self.main_model.layers_infer)
        total_pre_layers_num += sum(
            [len(previous_model.layers_infer) for previous_model in self.mtp_previous_draft_models]
        )
        self.layers_infer = [self.transformer_layer_infer_class(total_pre_layers_num, network_config=self.config)]
        assert self.layers_infer[0].compress_ratio == 0, "DeepSeek-V4 MTP draft layer must be SWA-only"
        return

    def _gen_special_model_input(self, token_num: int):
        return {
            "mtp_draft_input_hiddens": torch.randn(
                token_num,
                self.config["hc_mult"] * self.config["hidden_size"],
                dtype=self.data_type,
                device="cuda",
            )
        }

    def load_weights(self, weight_dict: dict):
        if weight_dict:
            return super().load_weights(weight_dict)

        index_file = os.path.join(self.weight_dir_, "model.safetensors.index.json")
        weight_map = utils.PetrelHelper.load_json(index_file)["weight_map"]
        candidate_files = sorted({file_ for key, file_ in weight_map.items() if key.startswith("mtp.0.")})

        loaded_key_count = 0
        desc = f"pid {os.getpid()} Loading DeepSeek-V4 MTP weights"
        for file_ in tqdm(candidate_files, total=len(candidate_files), desc=desc):
            weights = {}
            with safe_open(os.path.join(self.weight_dir_, file_), "pt", "cpu") as f:
                for key in f.keys():
                    if key.startswith("mtp.0."):
                        weights[key] = f.get_tensor(key)

            loaded_key_count += len(weights)
            self.pre_post_weight.load_hf_weights(weights)
            for layer in self.trans_layers_weight:
                layer.load_hf_weights(weights)
            del weights
            gc.collect()

        self.pre_post_weight.verify_load()
        for weight in self.trans_layers_weight:
            weight.verify_load()
        logger.info(f"loaded DeepSeek-V4 MTP weights: {loaded_key_count} tensors")
        return

    def autotune_layers(self):
        return 1
