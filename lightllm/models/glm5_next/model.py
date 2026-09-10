import json
import os

import torch
import triton

from lightllm.common.build_utils import repair_config
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
from lightllm.models.registry import ModelRegistry
from .attention import Glm5NextSparseAttBackend
from .cache_config import Glm5NextCacheConfig
from .kda_backend import KDALinearAttBackend
from .layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from .layer_weights.pre_and_post_layer_weight import Glm5NextPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Glm5NextTransformerLayerWeight
from .mem_manager import Glm5NextMemManager


@ModelRegistry(["glm5_next", "glm5_next_text"])
class Glm5NextTpPartModel(Deepseek3_2TpPartModel):
    pre_and_post_weight_class = Glm5NextPreAndPostLayerWeight
    transformer_weight_class = Glm5NextTransformerLayerWeight
    transformer_layer_infer_class = Glm5NextTransformerLayerInfer

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json")) as f:
            outer_config = json.load(f)
        self.config = dict(outer_config.get("text_config", outer_config))
        if "quantization_config" in outer_config:
            self.config["quantization_config"] = dict(outer_config["quantization_config"])
        self.config["quantization_config"].setdefault("scale_fmt", "ue8m0")
        self.config["autotune_layer_num"] = 4
        for names in (
            ["num_attention_heads", "n_head"],
            ["hidden_size", "n_embd", "n_embed"],
            ["num_hidden_layers", "n_layer"],
        ):
            repair_config(self.config, same_names=names)

    def _verify_params(self):
        super()._verify_params()
        args = self.args
        assert self.data_type == torch.bfloat16, "GLM-5.3 Flash currently requires bfloat16 activations"
        assert self.run_mode == "normal", "GLM-5.3 Flash v1 supports normal TP serving"
        assert args.dp == 1 and not args.enable_tpsp_mix_mode, "GLM-5.3 Flash v1 uses plain tensor parallelism"
        assert args.mtp_mode is None and args.mtp_step == 0, "GLM-5.3 Flash MTP is not implemented yet"
        assert not args.enable_ep_moe, "GLM-5.3 Flash v1 uses tensor-parallel MoE"
        assert not (
            args.enable_prefill_microbatch_overlap or args.enable_decode_microbatch_overlap
        ), "GLM-5.3 Flash mHC does not support microbatch overlap yet"
        assert not args.enable_prefill_cudagraph, "GLM-5.3 Flash v1 supports decode CUDA graphs"
        assert args.llm_kv_type in (None, "None"), "GLM-5.3 Flash v1 uses BF16 MLA KV with FP8 index keys"
        assert self.config["qk_rope_head_dim"] == 0 and self.config["kv_lora_rank"] == 512
        assert self.config["index_head_dim"] == 128 and self.config["index_kpool"] == 4

    def autotune_layers(self):
        return 4

    def _init_req_manager(self):
        self.linear_config = Glm5NextCacheConfig.from_model_config(self.config, self.args)
        self.req_manager = ReqManagerForMamba(
            self.max_req_num,
            max(self.batch_max_tokens or 0, self.max_seq_length or 0),
            None,
            linear_config=self.linear_config,
        )

    def _init_mem_manager(self):
        self.mem_manager = Glm5NextMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=1,
            head_dim=self.linear_config.full_att_head_dim,
            full_att_layer_num=self.linear_config.get_main_model_full_att_layer_num(),
            linear_config=self.linear_config,
            mem_fraction=self.mem_fraction,
        )

    def _init_att_backend(self):
        self.prefill_att_backend = Glm5NextSparseAttBackend(model=self)
        self.decode_att_backend = self.prefill_att_backend

    def _init_att_backend1(self):
        self.prefill_att_backend1 = KDALinearAttBackend(model=self)
        self.decode_att_backend1 = self.prefill_att_backend1

    def _init_custom(self):
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
        self._cos_cached = torch.empty((self.max_seq_length, 0), dtype=self.data_type, device="cuda")
        self._sin_cached = torch.empty_like(self._cos_cached)
