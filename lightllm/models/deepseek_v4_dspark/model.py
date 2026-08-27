import gc
import os
from typing import List

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

import lightllm.utils.petrel_helper as utils
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4_dspark.infer_struct import (
    DeepseekV4DSparkInferStateInfo,
)
from lightllm.models.deepseek_v4_dspark.layer_infer.post_layer_infer import (
    DeepseekV4DSparkPostLayerInfer,
)
from lightllm.models.deepseek_v4_dspark.layer_infer.pre_layer_infer import (
    DeepseekV4DSparkPreLayerInfer,
)
from lightllm.models.deepseek_v4_dspark.layer_infer.transformer_layer_infer import (
    DeepseekV4DSparkTransformerLayerInfer,
)
from lightllm.models.deepseek_v4_dspark.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4DSparkPreAndPostLayerWeight,
)
from lightllm.models.deepseek_v4_dspark.layer_weights.transformer_layer_weight import (
    DeepseekV4DSparkTransformerLayerWeight,
)
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@DraftModelRegistry(model_type="deepseek_v4", spec_modes="dspark")
class DeepseekV4DSparkModel(DeepseekV4TpPartModel):
    """Three-stage DSpark draft model backed by target-owned DeepSeek-V4 cache layers."""

    is_mtp_draft_model = True

    pre_and_post_weight_class = DeepseekV4DSparkPreAndPostLayerWeight
    transformer_weight_class = DeepseekV4DSparkTransformerLayerWeight
    pre_layer_infer_class = DeepseekV4DSparkPreLayerInfer
    post_layer_infer_class = DeepseekV4DSparkPostLayerInfer
    transformer_layer_infer_class = DeepseekV4DSparkTransformerLayerInfer
    infer_state_class = DeepseekV4DSparkInferStateInfo

    def __init__(self, kvargs: dict):
        self._pre_init(kvargs)
        super().__init__(kvargs)

    def _pre_init(self, kvargs: dict):
        self.main_model: TpPartBaseModel = kvargs.pop("main_model")
        self.mtp_previous_draft_models: List[TpPartBaseModel] = kvargs.pop(
            "mtp_previous_draft_models"
        )
        assert (
            not self.mtp_previous_draft_models
        ), "DeepSeek-V4 DSpark uses one parallel draft model"

    def _init_config(self):
        super()._init_config()
        draft_layer_num = len(self.config["compress_ratios"]) - self.config["n_layer"]
        assert draft_layer_num > 0
        assert (
            self.config["compress_ratios"][-draft_layer_num:] == [0] * draft_layer_num
        )
        assert len(self.config["dspark_target_layer_ids"]) == draft_layer_num
        self.config["dspark_layer_num"] = draft_layer_num

        # Aliases consumed by the shared block/Markov implementation.
        self.config["block_size"] = int(self.config["dspark_block_size"])
        self.config["mask_token_id"] = int(self.config["dspark_noise_token_id"])
        self.config["markov_rank"] = int(self.config["dspark_markov_rank"])
        self.config["markov_head_type"] = "vanilla"
        self.config["enable_confidence_head"] = True
        self.config["confidence_head_with_markov"] = True

    def _verify_params(self):
        super()._verify_params()
        assert (
            not self.enable_tpsp_mix_mode
        ), "DeepSeek-V4 DSpark draft model does not support TP-SP"
        assert self.args.mtp_step == self.config["dspark_block_size"], (
            f"DeepSeek-V4 DSpark requires --mtp_step {self.config['dspark_block_size']}, "
            f"got {self.args.mtp_step}"
        )

    def _init_quant(self):
        super()._init_quant()
        expert_quant_type = self.quant_cfg.get_quant_type(
            self.config["n_layer"] - 1, "fused_moe"
        )
        for layer_idx in range(
            self.config["n_layer"], len(self.config["compress_ratios"])
        ):
            self.quant_cfg.quant_cfg[layer_idx]["fused_moe"] = expert_quant_type

    def _init_weights(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type,
            network_config=self.config,
            quant_cfg=self.quant_cfg,
        )
        self.pre_post_weight.wte_weight_ = self.main_model.pre_post_weight.wte_weight_
        self.pre_post_weight.lm_head_weight_ = (
            self.main_model.pre_post_weight.lm_head_weight_
        )
        first_layer = self.config["n_layer"]
        self.trans_layers_weight = [
            self.transformer_weight_class(
                first_layer + stage_id,
                self.data_type,
                network_config=self.config,
                quant_cfg=self.quant_cfg,
            )
            for stage_id in range(self.config["dspark_layer_num"])
        ]

    def _init_req_manager(self):
        self.req_manager = self.main_model.req_manager

    def _init_mem_manager(self):
        self.mem_manager = self.main_model.mem_manager

    def _create_padded_decode_model_input(
        self, model_input: ModelInput, new_batch_size: int
    ):
        padded_input = super()._create_padded_decode_model_input(
            model_input, new_batch_size
        )
        scratch_pages = model_input.mtp_draft_swa_pages
        if padded_input is model_input or scratch_pages is None:
            return padded_input

        assert model_input.batch_size % self.block_size == 0
        assert new_batch_size % self.block_size == 0
        page_num = model_input.batch_size // self.block_size
        padded_page_num = new_batch_size // self.block_size
        assert scratch_pages.shape == (page_num,)
        # HOLD rows ignore the page value. Pad only the CUDA Graph input; the CPU
        # owner list must continue to contain exactly the pages that were allocated.
        padded_input.mtp_draft_swa_pages = F.pad(
            scratch_pages,
            (0, padded_page_num - page_num),
            mode="constant",
            value=0,
        )
        return padded_input

    def _init_infer_layer(self, start_layer_index=None):
        assert start_layer_index is None
        self.pre_infer = self.pre_layer_infer_class(network_config=self.config)
        self.post_infer = self.post_layer_infer_class(network_config=self.config)
        first_layer = self.config["n_layer"]
        self.layers_infer = [
            self.transformer_layer_infer_class(
                first_layer + stage_id, network_config=self.config
            )
            for stage_id in range(self.config["dspark_layer_num"])
        ]

    def _init_some_value(self):
        super()._init_some_value()
        self.layers_num = self.config["dspark_layer_num"]

    def _init_custom(self):
        self._freqs_cis_sliding = self.main_model._freqs_cis_sliding
        self._freqs_cis_compress = self.main_model._freqs_cis_compress
        self._cos_cached_sliding = self.main_model._cos_cached_sliding
        self._sin_cached_sliding = self.main_model._sin_cached_sliding
        self._cos_cached_compress = self.main_model._cos_cached_compress
        self._sin_cached_compress = self.main_model._sin_cached_compress
        self.dsv4_workspace = self.main_model.dsv4_workspace
        self.block_size = self.config["dspark_block_size"]
        self.mask_token_id = self.config["dspark_noise_token_id"]
        for layer in self.layers_infer:
            layer.freqs_cis = self._freqs_cis_sliding
            layer.cos_compress_table = self._cos_cached_compress
            layer.sin_compress_table = self._sin_cached_compress

    def _prepare_dsv4_slots(self, model_input: ModelInput) -> None:
        if not model_input.is_prefill:
            if model_input.mtp_draft_input_hiddens is not None:
                # Target verify already prepared these shared full/SWA slots.
                # This pass only commits target hiddens into the draft layers.
                model_input.mtp_decode_slot_prepare_indices = ()
            elif model_input.mem_indexes_cpu is not None:
                # Proposal-owned full slots live only until the next verify.
                # Put each request's complete block in a private scratch page;
                # this needs neither accepted-length D2H nor host seq metadata.
                (
                    model_input.mtp_draft_swa_pages_cpu,
                    model_input.mtp_draft_swa_pages,
                ) = self.mem_manager.alloc_dspark_swa_block(
                    mem_indexes=model_input.mem_indexes,
                    block_size=self.block_size,
                )
                model_input.mtp_decode_slot_prepare_indices = ()
        return super()._prepare_dsv4_slots(model_input)

    def _decode(self, model_input: ModelInput) -> ModelOutput:
        if model_input.mtp_draft_input_hiddens is None:
            return super()._decode(model_input)

        assert model_input.mtp_draft_input_hiddens.shape[0] == model_input.batch_size
        infer_state = self._create_inferstate(model_input)
        infer_state.position_ids = model_input.b_seq_len - 1

        hidden = self.pre_infer.context_forward(None, infer_state, self.pre_post_weight)
        for layer, layer_weight in zip(self.layers_infer, self.trans_layers_weight):
            hidden = layer.context_forward(hidden, infer_state, layer_weight)
        return ModelOutput(logits=hidden.new_empty((model_input.batch_size, 1)))

    def _gen_special_model_input(self, token_num: int):
        assert token_num % self.block_size == 0
        return {
            "mtp_draft_input_hiddens": None,
            # CUDA Graph capture must take the same direct scratch-slot branch
            # as runtime. HOLD rows never dereference the page value, but the
            # tensor gives the captured graph stable storage for replay copies.
            "mtp_draft_swa_pages": torch.zeros(
                token_num // self.block_size, dtype=torch.int32, device="cuda"
            ),
        }

    def _autotune_warmup(self):
        return

    def _init_padded_req(self):
        return

    def _init_prefill_cuda_graph(self):
        self.prefill_graph = None

    def load_weights(self, weight_dict: dict):
        if weight_dict:
            return super().load_weights(weight_dict)

        index_file = os.path.join(self.weight_dir_, "model.safetensors.index.json")
        assert utils.PetrelHelper.exists(
            index_file
        ), "DeepSeek-V4 DSpark requires model.safetensors.index.json"
        weight_map = utils.PetrelHelper.load_json(index_file)["weight_map"]
        candidate_files = sorted(
            {file_ for key, file_ in weight_map.items() if key.startswith("mtp.")}
        )
        assert (
            candidate_files
        ), "DeepSeek-V4 DSpark weights with prefix mtp.* were not found"

        loaded_key_count = 0
        desc = f"pid {os.getpid()} Loading DeepSeek-V4 DSpark weights"
        for file_ in tqdm(candidate_files, total=len(candidate_files), desc=desc):
            weights = {}
            with safe_open(os.path.join(self.weight_dir_, file_), "pt", "cpu") as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        weights[key] = f.get_tensor(key)

            loaded_key_count += len(weights)
            self.pre_post_weight.load_hf_weights(weights)
            for layer in self.trans_layers_weight:
                layer.load_hf_weights(weights)
            del weights
            gc.collect()

        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        logger.info("loaded DeepSeek-V4 DSpark weights: %d tensors", loaded_key_count)
