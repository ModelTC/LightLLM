from __future__ import annotations

from typing import TYPE_CHECKING
import dataclasses

from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num
from lightllm.utils.torch_dtype_utils import get_torch_dtype

from lightllm.models.qwen3next.att_resource_config import Qwen3NextResourceConfig

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.att_resources.att_tail import AttentionTailState
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


@dataclasses.dataclass
class Glm5NextResourceConfig(Qwen3NextResourceConfig):
    """Replicated MLA/index KV plus TP-sharded KDA checkpoints."""

    # Indexer 池化步长，单位为原始 token；同时决定 K-pool tail 的基准 ring 长度和恢复边界。
    index_kpool: int = 4
    # 单个 indexer key 的特征维度；原始 key/gate tail 的 channel 宽度为该值的两倍。
    index_head_dim: int = 128

    # 当前 packed MLA 行除 latent 外预留的字节数，不是 dtype 元素数。
    # 包含 128 字节量化 indexer key、4 字节 scale，并向上补齐到 16 字节边界。
    INDEX_PADDING_BYTES = 144

    @classmethod
    def from_model_config(cls, config: dict[str, object], args: StartArgs | EasyDict, *, include_draft: bool = True):
        config = config.get("text_config", config)
        linear = config["linear_attn_config"]
        tp = args.tp // args.dp
        assert linear["num_heads"] % tp == 0
        layers = config["num_hidden_layers"]
        assert config["layer_types"] == [
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)
        ]
        dtype = get_torch_dtype(args.data_type)
        # Raw index keys and compression scores only live in request tails.
        packed_dim = config["kv_lora_rank"] + cls.INDEX_PADDING_BYTES // dtype.itemsize
        return cls(
            tp_world_size=tp,
            full_att_all_num_kv_heads=1,
            full_att_dtype=dtype,
            full_att_num_kv_heads=1,
            full_att_head_dim=packed_dim,
            global_linear_k_heads=linear["num_heads"],
            global_linear_v_heads=linear["num_heads"],
            num_linear_k_heads=linear["num_heads"] // tp,
            num_linear_v_heads=linear["num_heads"] // tp,
            head_linear_k_dim=linear["head_dim"],
            head_linear_v_dim=linear["head_dim"],
            conv_kernel_size=linear["short_conv_kernel_size"],
            linear_layer_num=len(linear["kda_layers"]),
            conv_state_dtype=dtype,
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=4,
            all_layer_num=layers,
            draft_full_att_kv_layer_num=get_added_mtp_kv_layer_num()
            if include_draft and args.mtp_mode is not None
            else 0,
            index_kpool=config["index_kpool"],
            index_head_dim=config["index_head_dim"],
        )


def create_model_att_tail_states(
    model_class: type[TpPartBaseModel], model_config: dict[str, object], args: StartArgs | EasyDict, *,
    model_logical_layer_ids: tuple[int, ...], request_capacity: int,
    tp_rank: int, tp_world_size: int, device: str,
) -> list[AttentionTailState]:
    """由模型配置创建 K-pool tail 列表，不读取模型实例或注册管理器。"""
    from lightllm.common.att_resources.att_tail import create_kpool_state, KPoolStateConfig
    from lightllm.common.basemodel.basemodel_utils import get_full_attention_options, get_resource_config

    config = get_resource_config(model_config, model_class=model_class)
    options = get_full_attention_options(
        config, args, model_class=model_class, model_logical_layer_ids=model_logical_layer_ids,
    )
    if not options["logical_layer_groups"]:
        return []
    assert args.linear_att_hash_page_size % config["index_kpool"] == 0, "GLM cache pages must align to index_kpool"
    state = create_kpool_state(
        config=KPoolStateConfig(
            logical_layer_ids=tuple(
                logical_layer_id for logical_layer_group in options["logical_layer_groups"]
                for logical_layer_id in logical_layer_group
            ),
            pool_size=config["index_kpool"], head_dim=config["index_head_dim"],
            dtype=get_torch_dtype(args.data_type), tp_rank=tp_rank, tp_world_size=tp_world_size,
        ),
        request_capacity=request_capacity, mtp_step=args.mtp_step, device=device,
    )
    return [state]
