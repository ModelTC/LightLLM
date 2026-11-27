from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerFast

from lightllm.distributed.communication_op import dist_group_manager
from lightllm.models.deepseek_v4.config import create_swa_config
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.layer_weights.pre_and_post_layer_weight import DeepseekV4EmbeddingAndHeadWeight
from lightllm.models.deepseek_v4.workspace import DeepseekV4Workspace
from .config import create_compressed_configs, create_compressor_configs
from .engram import EngramHasher
from .infer_struct import DeepseekV41InferStateInfo
from .layer_infer.post_layer_infer import DeepseekV41PostLayerInfer
from .layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer
from .layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.common.attention_state.bounded_att.base_state import BoundedAttentionState
    from lightllm.common.attention_state.full_att.base_state import FullAttentionState
    from lightllm.server.core.objs.start_args_type import StartArgs


class DeepseekV41TpPartModel(DeepseekV4TpPartModel):
    # The last FFN supplies the head mixing coefficients.
    pre_and_post_weight_class = DeepseekV4EmbeddingAndHeadWeight
    transformer_weight_class = DeepseekV41TransformerLayerWeight
    transformer_layer_infer_class = DeepseekV41TransformerLayerInfer
    post_layer_infer_class = DeepseekV41PostLayerInfer
    infer_state_class = DeepseekV41InferStateInfo

    @classmethod
    def normalize_model_config(cls, model_config):
        config = super().normalize_model_config(model_config)
        config.update({key: value for key, value in model_config.items() if key != "text_config"})
        config["num_hash_layers"] = 0
        config["vision_n_layers"] = (config.get("vision_config") or {}).get("num_hidden_layers", 0)
        config.setdefault("expert_dtype", config.get("quantization_config", {}).get("expert_dtype"))
        return config

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: StartArgs | EasyDict,
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list[FullAttentionState]:
        from lightllm.common.attention_state.full_att.compressed_att.create_compressed_att_state import (
            create_compressed_att_state,
        )

        assert args.llm_kv_type in ("None", "fp8kv_dsa"), f"unsupported V4.1 KV encoding: {args.llm_kv_type}"
        return [
            create_compressed_att_state(config=config, page_size=args.page_size, device="meta")
            for config in create_compressed_configs(model_config, model_logical_layer_ids, tp_rank, tp_world_size)
        ]

    @classmethod
    def create_bounded_att_states(
        cls,
        model_config: dict[str, object],
        args: StartArgs | EasyDict,
        model_logical_layer_ids: tuple[int, ...],
        request_capacity: int = 1,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        device: str = "meta",
    ) -> list[BoundedAttentionState]:
        from lightllm.common.attention_state.bounded_att import (
            create_sliding_window_state,
            create_window_compressor_state,
        )

        swa_config = create_swa_config(model_config, model_logical_layer_ids, tp_rank, tp_world_size)
        compressor_configs = create_compressor_configs(model_config, model_logical_layer_ids, tp_rank, tp_world_size)
        states = [
            create_sliding_window_state(
                config=swa_config, request_capacity=request_capacity, mtp_step=args.mtp_step, device=device
            )
        ]
        for config in compressor_configs:
            states.append(
                create_window_compressor_state(
                    config=config, request_capacity=request_capacity, mtp_step=args.mtp_step, device=device
                )
            )
        return states

    def _verify_params(self):
        super()._verify_params()
        if self.args.mtp_mode is not None:
            raise ValueError("DeepSeek-V4.1 does not support speculative decoding yet")
        if (
            self.enable_tpsp_mix_mode
            or self.args.enable_prefill_microbatch_overlap
            or self.args.enable_decode_microbatch_overlap
        ):
            raise ValueError("DeepSeek-V4.1 does not support TP-SP or microbatch overlap")
        if self.args.enable_prefill_cudagraph:
            raise ValueError("DeepSeek-V4.1 requires eager prefill")
        assert self.config["candidate_block_size"] == 8, "the V4.1 indexer selects blocks of eight keys"

    def _init_custom(self):
        self._init_to_get_rotary()
        # V4.1 uses this workspace for FlashMLA Q. Its C1/C2 compressors need neither V4 scratch buffer.
        self.dsv4_workspace = DeepseekV4Workspace(self)
        dist_group_manager.new_deepep_group(
            n_routed_experts=self.config["n_routed_experts"],
            hidden_size=self.config["hidden_size"],
            expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
            num_experts_per_tok=self.config["num_experts_per_tok"],
            moe_intermediate_size=self.config["moe_intermediate_size"],
        )

    def init_runtime(self):
        # History uses the same physical token slots as full attention, including the HOLD page.
        allocator = self.att_state_manager.full_att_allocator
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.weight_dir_)
        self.engram_hasher = EngramHasher(
            self.config, tokenizer, num_kv_token_slots=allocator.size + allocator.page_size
        )
        self.token_lookback_depth = self.engram_hasher.max_ngram_size - 1
        super().init_runtime()

    def forward(self, model_input):
        if model_input.token_lookback:
            self.engram_hasher.restore_token_history(model_input.token_lookback, self.req_manager.req_to_token_indexs)
        return super().forward(model_input)


class DeepseekV41VisionTpPartModel(DeepseekV41TpPartModel):
    has_vision = True
