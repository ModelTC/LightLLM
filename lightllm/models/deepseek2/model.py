from typing import TYPE_CHECKING
import torch
from typing import final
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.distributed.communication_op import dist_group_manager
from lightllm.common.basemodel.attention import get_mla_decode_att_backend_class, get_mla_prefill_att_backend_class
from lightllm.common.execution.model_adapter import RecipeModelAdapter
from lightllm.common.execution.specs import MLAAttentionSpec, StateSpec
from lightllm.models.llama.model import LlamaModelAdapter
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
from copy import deepcopy

logger = init_logger(__name__)


if TYPE_CHECKING:
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.attention_state.full_att import FullAttentionState


class Deepseek2ModelAdapter(RecipeModelAdapter):
    moe_group_score_topk = 1
    pre_and_post_weight_class = LlamaModelAdapter.pre_and_post_weight_class
    pre_layer_infer_class = LlamaModelAdapter.pre_layer_infer_class
    post_layer_infer_class = LlamaModelAdapter.post_layer_infer_class
    transformer_weight_class = Deepseek2TransformerLayerWeight
    transformer_layer_infer_class = Deepseek2TransformerLayerInfer

    def moe_specs(self):
        from lightllm.common.execution.model_adapter import grouped_moe_specs

        return grouped_moe_specs(self.config, self.dtype, group_score_topk=self.moe_group_score_topk)

    def __init__(self, config, *, role, dtype, tp_world_size=1, layer_ids=None):
        self.config, self.dtype = deepcopy(config), dtype
        if config.get("model_type") in ("deepseek_v3", "deepseek_v32"):
            self.moe_group_score_topk = 2
        heads, depth = config["num_attention_heads"], config["num_hidden_layers"]
        if depth <= 0 or tp_world_size <= 0 or heads <= 0 or heads % tp_world_size:
            raise ValueError("MLA requires layers and whole query-head TP shards")
        scale = (config["qk_nope_head_dim"] + config["qk_rope_head_dim"]) ** -0.5
        rope = config.get("rope_scaling") or {}
        if rope.get("mscale_all_dim", 0):
            mscale = get_deepseek_mscale(rope["factor"], rope["mscale_all_dim"])
            scale = scale * mscale * mscale
        spec = MLAAttentionSpec(
            heads // tp_world_size,
            config["kv_lora_rank"],
            config["qk_nope_head_dim"],
            config["qk_rope_head_dim"],
            config["v_head_dim"],
            dtype,
            scale,
        )
        self.layer_ids = tuple(range(depth)) if layer_ids is None else tuple(layer_ids)
        if len(self.layer_ids) != depth or len(set(self.layer_ids)) != depth or min(self.layer_ids) < 0:
            raise ValueError("MLA execution layer IDs must uniquely describe every model layer")
        self._attention = {i: spec for i in self.layer_ids}
        self._states = tuple(
            StateSpec(role, f"mla.{i}", i, (i,), (spec.latent_dim, spec.rope_dim)) for i in self.layer_ids
        )

    def validate_layer(self, layer):
        spec = self._attention[layer.layer_num_]
        if (
            layer.tp_q_head_num_,
            layer.kv_lora_rank,
            layer.qk_nope_head_dim,
            layer.qk_rope_head_dim,
            layer.v_head_dim,
            layer.softmax_scale,
        ) != (spec.q_heads, spec.latent_dim, spec.nope_dim, spec.rope_dim, spec.value_dim, spec.scale):
            raise ValueError("MLA recipe projections disagree with the declared semantics")


class Deepseek2TpPartModel(LlamaTpPartModel):
    model_adapter_class = Deepseek2ModelAdapter
    # weight class
    transformer_weight_class = Deepseek2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek2TransformerLayerInfer

    # infer state class
    infer_state_class = Deepseek2InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_att_backend(self):
        self.prefill_att_backend = get_mla_prefill_att_backend_class(index=0)(model=self)
        self.decode_att_backend = get_mla_decode_att_backend_class(index=0)(model=self)
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 0

        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.v_head_dim = self.config.get("v_head_dim", self.qk_nope_head_dim)
        self.head_dim_ = self.kv_lora_rank + self.qk_rope_head_dim

    def _init_custom(self):
        self._init_to_get_yarn_rotary()
        dist_group_manager.new_deepep_group(
            n_routed_experts=self.config["n_routed_experts"],
            hidden_size=self.config["hidden_size"],
            expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
            num_experts_per_tok=self.config.get("num_experts_per_tok", 1),
            moe_intermediate_size=self.config.get("moe_intermediate_size", self.config.get("intermediate_size")),
        )

    def _verify_params(self):
        return super()._verify_params()

    @classmethod
    def create_full_att_states(
        cls,
        model_config: dict[str, object],
        args: "StartArgs | EasyDict",
        *,
        model_logical_layer_ids: tuple[int, ...],
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> list["FullAttentionState"]:
        """DeepSeek MLA 保存 latent 与 RoPE，TP 副本规则由 MLA 创建入口处理。"""
        from lightllm.common.attention_state.full_att import MLAConfig, create_mla_state
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        config = cls.normalize_model_config(model_config)
        logical_layer_groups = tuple((layer_id,) for layer_id in model_logical_layer_ids)
        if not logical_layer_groups:
            return []
        state = create_mla_state(
            config=MLAConfig(
                logical_layer_groups=logical_layer_groups,
                latent_dim=config["kv_lora_rank"] + config["qk_rope_head_dim"],
                latent_dtype=get_torch_dtype(args.data_type),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            page_size=args.page_size,
            device="meta",
        )
        return [state]

    def _init_to_get_yarn_rotary(self):
        from lightllm.models.llama.yarn_rotary_utils import find_correction_range, linear_ramp_mask, get_deepseek_mscale

        dim = self.qk_rope_head_dim
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        base = self.config.get("rope_theta", 10000.0)
        rope_scaling = self.config.get("rope_scaling") or {}
        scale = rope_scaling.get("factor", 1.0)
        mscale = rope_scaling.get("mscale", 1)
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
        extrapolation_factor = 1.0
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)

        pos_freqs = base ** (torch.arange(0, dim, 2).float().cuda() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, dim // 2).float().cuda()
        ) * extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        _mscale = float(
            get_deepseek_mscale(scale, mscale) / get_deepseek_mscale(scale, mscale_all_dim)
        )  # Get n-d magnitude scaling corrected for interpolation

        # Build here to make `torch.jit.trace` work.
        max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_seq_len_cached, device="cuda", dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self._cos_cached = (freqs.cos() * _mscale).to(self.data_type).cuda()
        self._sin_cached = (freqs.sin() * _mscale).to(self.data_type).cuda()

        return
