from __future__ import annotations

from typing import TYPE_CHECKING
import dataclasses

from lightllm.utils.torch_dtype_utils import get_torch_dtype

from lightllm.models.qwen3next.config import Qwen3NextResourceConfig

if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


@dataclasses.dataclass
class Glm5NextResourceConfig(Qwen3NextResourceConfig):
    """Replicated MLA/index KV plus TP-sharded KDA checkpoints."""

    # Indexer 池化步长，单位为原始 token；同时决定 K-pool tail 的基准 ring 长度和恢复边界。
    index_kpool: int = 4
    # 单个 indexer key 的特征维度；原始 key/gate tail 的 channel 宽度为该值的两倍。
    index_head_dim: int = 128

    @classmethod
    def from_model_config(cls, config: dict[str, object], args: StartArgs | EasyDict) -> "Glm5NextResourceConfig":
        config = config.get("text_config", config)
        linear = config["linear_attn_config"]
        tp = args.tp // args.dp
        assert linear["num_heads"] % tp == 0
        layers = config["num_hidden_layers"]
        assert config["layer_types"] == [
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)
        ]
        dtype = get_torch_dtype(args.data_type)
        return cls(
            tp_world_size=tp,
            full_att_all_num_kv_heads=1,
            full_att_head_dim=config["kv_lora_rank"],
            global_linear_k_heads=linear["num_heads"],
            global_linear_v_heads=linear["num_heads"],
            head_linear_k_dim=linear["head_dim"],
            head_linear_v_dim=linear["head_dim"],
            conv_kernel_size=linear["short_conv_kernel_size"],
            conv_state_dtype=dtype,
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=4,
            all_layer_num=layers,
            index_kpool=config["index_kpool"],
            index_head_dim=config["index_head_dim"],
        )
