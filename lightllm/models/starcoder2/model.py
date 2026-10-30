from typing import TYPE_CHECKING
import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.starcoder2.layer_weights.pre_and_post_layer_weight import Starcoder2PreAndPostLayerWeight
from lightllm.models.starcoder2.layer_weights.transformer_layer_weight import Starcoder2TransformerLayerWeight
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.starcoder2.layer_infer.transformer_layer_infer import Starcoder2TransformerLayerInfer
from lightllm.models.bloom.layer_infer.post_layer_infer import BloomPostLayerInfer

from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel import TpPartBaseModel


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.att_resources.full_att import FullAttentionState


class Starcoder2TpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = Starcoder2PreAndPostLayerWeight
    transformer_weight_class = Starcoder2TransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    transformer_layer_infer_class = Starcoder2TransformerLayerInfer
    post_layer_infer_class = BloomPostLayerInfer
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    @classmethod
    def _create_full_att_states(
        cls, model_config: dict[str, object], args: "StartArgs | EasyDict", *,
        model_logical_layer_ids: tuple[int, ...], tp_rank: int = 0, tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """按当前模型的方案创建 full-attention 定义，不分配 KV 容量或注册管理器。"""
        from lightllm.common.basemodel.basemodel_utils import create_full_attention_definition

        state = create_full_attention_definition(
            model_class=cls, model_config=model_config, args=args,
            model_logical_layer_ids=model_logical_layer_ids, tp_rank=tp_rank, tp_world_size=tp_world_size,
        )
        return [] if state is None else [state]

    def _init_config(self):
        super()._init_config()
        repair_config(self.config, same_names=["norm_epsilon", "rms_norm_eps", "layer_norm_epsilon"])
        if self.config["sliding_window"] is None:
            self.config["sliding_window"] = self.max_total_token_num
        # rename key [SYM: to be confirmed]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "mistral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        return

    def _init_some_value(self):
        super()._init_some_value()
        return

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_)
        )
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return
