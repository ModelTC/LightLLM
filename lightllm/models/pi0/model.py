from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from safetensors import safe_open

from lightllm.common.basemodel.attention import DenseSplitPrefillAttBackend
from lightllm.common.basemodel.cuda_graph import CudaGraphRunner
from lightllm.common.quantization import Quantcfg
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.gemma4.layer_infer.pre_layer_infer import Gemma4PreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_infer import (
    Pi0ActionPostLayerInfer,
    Pi0ActionPreLayerInfer,
    Pi0ActionTransformerLayerInfer,
)
from lightllm.models.pi0.layer_weights import (
    Pi0ActionPreAndPostLayerWeight,
    Pi0ActionTransformerLayerWeight,
    Pi0VLMPreAndPostLayerWeight,
    Pi0VLMTransformerLayerWeight,
)
from lightllm.models.pi0.preprocessing import (
    Pi0PrePostProcessor,
    load_pi_action_config,
)
from lightllm.models.registry import GenerationModelRegistry, ModelRegistry
from lightllm.utils.config_utils import (
    get_pi_config,
    with_pi_config_overrides,
)
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.torch_dtype_utils import get_torch_dtype


# Pi0 and Pi0.5 both use the fixed PaliGemma vocabulary from OpenPI.  LeRobot's
# policy config omits this model-internal value, so keep it with the model
# definition instead of inspecting the checkpoint in generic config code.
PI_VOCAB_SIZE = 257_152


def _load_matching_weights(checkpoint, target, predicate) -> None:
    """Feed matching tensors through the normal BaseWeight loading API."""

    for name in checkpoint.keys():
        if not predicate(name):
            continue
        value = checkpoint.get_tensor(name)
        target.load_hf_weights({name: value})
        del value


@ModelRegistry("pi0", is_multimodal=True)
class Pi0VLMModel(Gemma_2bTpPartModel):
    """PaliGemma prefix model implemented by the normal BaseModel pipeline."""

    pre_and_post_weight_class = Pi0VLMPreAndPostLayerWeight
    transformer_weight_class = Pi0VLMTransformerLayerWeight
    pre_layer_infer_class = Gemma4PreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Gemma_2bTransformerLayerInfer

    # The only attention-policy difference from an ordinary Gemma prefill.
    prefill_causal = False

    def __init__(self, kvargs: dict):
        self.vla_config = kvargs.get("vla_config")
        if self.vla_config is None:
            start_args = get_env_start_args()
            self.vla_config = getattr(start_args, "generation_runtime_config", None)
            if self.vla_config is None:
                self.vla_config = get_pi_config(
                    kvargs["weight_dir"],
                    dtype=kvargs.get("data_type"),
                )
        self.vla_config = with_pi_config_overrides(self.vla_config)
        super().__init__(kvargs)

    def _init_config(self):
        config = self.vla_config
        self.config = {
            "model_type": "gemma",
            "hidden_size": config["vlm_hidden_size"],
            "n_embed": config["vlm_hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "n_layer": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "intermediate_size": config["vlm_intermediate_size"],
            "vocab_size": PI_VOCAB_SIZE,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": self.max_seq_length,
            "tie_word_embeddings": True,
        }

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = max(
            self.config["num_key_value_heads"] // self.tp_world_size_, 1
        )
        self.tp_v_head_num_ = self.tp_k_head_num_

    def _init_custom(self):
        self.use_ieee_fp32_attention = self.data_type is torch.float32
        super()._init_custom()

    def _load_hf_weights(self):
        with safe_open(
            self.vla_config["checkpoint_path"], framework="pt", device="cpu"
        ) as checkpoint:
            prepost_names = {
                "paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.paligemma.model.language_model.norm.weight",
            }
            _load_matching_weights(
                checkpoint, self.pre_post_weight, lambda name: name in prepost_names
            )
            for layer_index, layer_weight in enumerate(self.trans_layers_weight):
                prefix = (
                    "paligemma_with_expert.paligemma.model.language_model.layers."
                    f"{layer_index}."
                )
                _load_matching_weights(
                    checkpoint,
                    layer_weight,
                    lambda name, prefix=prefix: name.startswith(prefix),
                )
        self.pre_post_weight.verify_load()
        for layer_weight in self.trans_layers_weight:
            layer_weight.verify_load()


@dataclass
class Pi0ActionOutput:
    actions: torch.Tensor
    timing_events: tuple[torch.cuda.Event, torch.cuda.Event] | None

    def resolve_timing(self) -> float | None:
        if self.timing_events is None:
            return None
        start_event, end_event = self.timing_events
        elapsed_ms = start_event.elapsed_time(end_event)
        self.timing_events = None
        return elapsed_ms


@GenerationModelRegistry("pi0", output_modality="action")
class Pi0ActionExpertModel:
    """Pi0 diffusion controller with a continuous-state action suffix."""

    generation_config_loader = staticmethod(load_pi_action_config)
    generation_processor_class = Pi0PrePostProcessor

    pre_layer_infer_class = Pi0ActionPreLayerInfer
    post_layer_infer_class = Pi0ActionPostLayerInfer
    transformer_layer_infer_class = Pi0ActionTransformerLayerInfer
    pre_and_post_weight_class = Pi0ActionPreAndPostLayerWeight
    transformer_weight_class = Pi0ActionTransformerLayerWeight
    first_suffix_token_isolated = True
    action_weight_prefixes = (
        "action_in_proj.",
        "action_out_proj.",
        "action_time_mlp_",
        "state_proj.",
        "paligemma_with_expert.gemma_expert.model.norm.",
    )

    def __init__(
        self,
        config: dict,
        *,
        max_seq_length: int,
        dist_group=None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
        quant_type: str = "none",
        quant_cfg_path: str | None = None,
        enable_cuda_graph: bool = False,
    ):
        self.vla_config = with_pi_config_overrides(config)
        self.device = torch.device(device)
        self.data_type = dtype or get_torch_dtype(config["dtype"])
        self.dist_group = dist_group
        self.layers_num = config["num_hidden_layers"]
        self.max_seq_length = int(max_seq_length)
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        self.enable_cuda_graph = (
            enable_cuda_graph
            and self.device.type == "cuda"
            and self.data_type in (torch.float16, torch.bfloat16)
        )
        self.graph = CudaGraphRunner() if self.enable_cuda_graph else None
        self.config = {
            "hidden_size": config["expert_hidden_size"],
            "n_embed": config["expert_hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "n_layer": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "intermediate_size": config["expert_intermediate_size"],
            "vocab_size": PI_VOCAB_SIZE,
            "rms_norm_eps": 1e-6,
            "max_action_dim": config["max_action_dim"],
            "max_state_dim": config["max_state_dim"],
        }
        self.quant_cfg = Quantcfg(
            self.config,
            quant_type=quant_type or "none",
            custom_cfg_path=quant_cfg_path,
        )
        self.pre_infer = self.pre_layer_infer_class(self.config)
        self.post_infer = self.post_layer_infer_class(self.config)
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.data_type, self.config, self.quant_cfg
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                layer_index,
                self.data_type,
                self.config,
                self.quant_cfg,
            )
            for layer_index in range(config["num_hidden_layers"])
        ]
        self.layers_infer = [
            self.transformer_layer_infer_class(layer_index, self.config)
            for layer_index in range(config["num_hidden_layers"])
        ]
        self.prefill_att_backend = DenseSplitPrefillAttBackend(self)
        self._load_weights()
        self._init_rope_cache()
        self._denoise_constants = self._init_denoise_constants(
            config["num_denoise_steps"]
        )

    def _load_weights(self):
        with safe_open(
            self.vla_config["checkpoint_path"], framework="pt", device="cpu"
        ) as checkpoint:
            action_names = {
                name
                for name in checkpoint.keys()
                if name.startswith(self.action_weight_prefixes)
            }
            _load_matching_weights(
                checkpoint, self.pre_post_weight, lambda name: name in action_names
            )
            for layer_index, layer_weight in enumerate(self.trans_layers_weight):
                prefix = (
                    "paligemma_with_expert.gemma_expert.model.layers." f"{layer_index}."
                )
                _load_matching_weights(
                    checkpoint,
                    layer_weight,
                    lambda name, prefix=prefix: name.startswith(prefix),
                )
        self.pre_post_weight.verify_load()
        for layer_weight in self.trans_layers_weight:
            layer_weight.verify_load()

    def _init_rope_cache(self):
        positions = torch.arange(
            self.max_seq_length,
            dtype=torch.float32,
            device=self.device,
        )
        exponents = (
            torch.arange(
                0,
                self.vla_config["head_dim"],
                2,
                dtype=torch.float32,
                device=self.device,
            )
            / self.vla_config["head_dim"]
        )
        inverse_frequency = 1.0 / (10000.0**exponents)
        radians = torch.outer(positions, inverse_frequency)
        self._cos_cached = radians.cos().to(self.data_type)
        self._sin_cached = radians.sin().to(self.data_type)

    def _init_denoise_constants(
        self, max_steps: int
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Build every allowed flow schedule once, alongside the RoPE cache."""

        hidden_size = self.vla_config["expert_hidden_size"]
        fractions = torch.linspace(
            0.0,
            1.0,
            hidden_size // 2,
            dtype=torch.float64,
            device=self.device,
        )
        periods = (
            self.vla_config["min_period"]
            * (self.vla_config["max_period"] / self.vla_config["min_period"])
            ** fractions
        )
        frequencies = 2.0 * math.pi / periods
        constants = {}
        for num_steps in range(1, max_steps + 1):
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=self.device)
            timesteps = (
                1.0
                + torch.arange(num_steps, dtype=torch.float32, device=self.device) * dt
            )
            radians = timesteps.to(torch.float64)[:, None] * frequencies[None, :]
            time_embeddings = torch.cat(
                (torch.sin(radians), torch.cos(radians)), dim=-1
            ).to(self.data_type)
            constants[num_steps] = (time_embeddings, dt)
        return constants

    def _get_denoise_constants(
        self, num_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up a flow schedule materialized during model initialization."""

        constants = self._denoise_constants.get(num_steps)
        if constants is None:
            raise ValueError(
                f"num_steps must be in [1, {len(self._denoise_constants)}]"
            )
        return constants

    def _pad_state(self, state: torch.Tensor) -> torch.Tensor:
        """Match a robot state to OpenPI's fixed checkpoint state width."""

        target = self.vla_config["max_state_dim"]
        if state.shape[-1] > target:
            raise ValueError(
                f"state dimension {state.shape[-1]} exceeds checkpoint dimension {target}"
            )
        if state.shape[-1] == target:
            return state
        return torch.nn.functional.pad(state, (0, target - state.shape[-1]))

    def _prepare_state(self, state: torch.Tensor | None) -> torch.Tensor:
        """Validate Pi0's required state once, before the denoise hot path."""

        if state is None:
            raise ValueError("pi0 action sampling requires continuous state")
        return self._pad_state(state)

    def _suffix_length(self, action_horizon: int) -> int:
        # Pi0 prepends one continuous-state token to the noisy action tokens.
        return action_horizon + 1

    def _validate_action_metadata(
        self,
        *,
        prefix_kv,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        action_horizon: int,
    ) -> None:
        if noise.ndim != 3 or noise.shape[1] != action_horizon:
            raise ValueError(
                "noise must have shape [batch, action_horizon, max_action_dim]"
            )
        batch_size = noise.shape[0]
        if noise.shape[-1] != self.vla_config["max_action_dim"]:
            raise ValueError(
                f"noise dimension must equal checkpoint dimension {self.vla_config['max_action_dim']}"
            )
        if prefix_kv.batch_size != batch_size:
            raise ValueError("one dense prefix is required per action row")
        suffix_length = self._suffix_length(action_horizon)
        if state is not None and (state.ndim != 2 or state.shape[0] != batch_size):
            raise ValueError("state must have shape [batch, state_dim]")
        # Shape checks stay host-only; the producer already validated every
        # length before the dense gather.  The local page-table width is the
        # largest prefix in this batch.
        if prefix_kv.page_table.shape[1] + suffix_length > self.max_seq_length:
            raise ValueError(
                "action prefix and positions exceed the configured sequence length"
            )
        return

    def _create_infer_state(
        self,
        *,
        prefix_kv,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        num_steps: int,
    ) -> Pi0ActionInferStateInfo:
        _, action_horizon = noise.shape[:2]
        suffix_length = self._suffix_length(action_horizon)
        batch_size = noise.shape[0]
        position_offsets = torch.arange(
            suffix_length, dtype=torch.int32, device=self.device
        )
        position_ids = prefix_kv.position_offsets[:, None] + position_offsets[None, :]
        flat_positions = position_ids.reshape(-1).long()
        position_cos = torch.index_select(self._cos_cached, 0, flat_positions)
        position_sin = torch.index_select(self._sin_cached, 0, flat_positions)
        time_embeddings, dt = self._get_denoise_constants(num_steps)
        infer_state = Pi0ActionInferStateInfo(
            batch_size=batch_size,
            suffix_length=suffix_length,
            prefix_kv=prefix_kv,
            prefix_lengths=prefix_kv.prefix_lengths,
            position_cos=position_cos,
            position_sin=position_sin,
            state=state,
            noise=noise,
            action_horizon=action_horizon,
            time_embeddings=time_embeddings,
            dt=dt,
            first_suffix_token_isolated=self.first_suffix_token_isolated,
            dist_group=self.dist_group,
        )
        infer_state.prefill_att_state = (
            self.prefill_att_backend.create_att_prefill_state(infer_state=infer_state)
        )
        return infer_state

    def _denoise_step(
        self,
        infer_state: Pi0ActionInferStateInfo,
        noisy_actions: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.pre_infer.context_forward(
            infer_state.state, noisy_actions, time_embedding, self.pre_post_weight
        )
        for layer, weight in zip(
            self.layers_infer, self.trans_layers_weight, strict=True
        ):
            hidden_states = layer.context_forward(
                hidden_states,
                infer_state,
                weight,
            )
        return self.post_infer.token_forward(
            hidden_states,
            infer_state.action_horizon,
            self.pre_post_weight,
        )

    def _context_forward(self, infer_state: Pi0ActionInferStateInfo) -> torch.Tensor:
        infer_state.prefill_att_state.init_state()
        actions = infer_state.noise
        for time_embedding in infer_state.time_embeddings:
            velocity = self._denoise_step(
                infer_state,
                actions,
                time_embedding.expand(actions.shape[0], -1),
            )
            actions = actions + infer_state.dt * velocity
        return actions

    @torch.no_grad()
    def forward(
        self,
        *,
        prefix_kv,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        num_steps: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
    ) -> Pi0ActionOutput:
        num_steps = num_steps or self.vla_config["num_denoise_steps"]
        action_dim = action_dim or noise.shape[-1]
        action_horizon = action_horizon or noise.shape[-2]
        if action_dim > self.vla_config["max_action_dim"]:
            raise ValueError("requested action_dim exceeds checkpoint maximum")
        if prefix_kv.batch_size != noise.shape[0]:
            raise ValueError("prefix and noise batch sizes differ")
        # OpenPI samples every checkpoint action dimension from N(0, 1), then
        # crops the generated result to the robot-specific action width. State
        # is real input data, so only state uses OpenPI's zero-padding transform.
        state = self._prepare_state(state)
        self._validate_action_metadata(
            prefix_kv=prefix_kv,
            state=state,
            noise=noise,
            action_horizon=action_horizon,
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        graph_key = (
            tuple(noise.shape),
            None if state is None else tuple(state.shape),
            num_steps,
            prefix_kv.layout_key,
        )
        # The runner retains the first infer state for each graph key. Force
        # ownership only for that capture; steady replay keeps one input copy.
        capture_owns_inputs = (
            self.enable_cuda_graph
            and self.graph is not None
            and self.graph.needs_capture(graph_key)
        )
        infer_state = self._create_infer_state(
            prefix_kv=prefix_kv,
            state=None
            if state is None
            else state.to(
                self.device,
                dtype=self.data_type,
                non_blocking=True,
                copy=capture_owns_inputs,
            ),
            noise=noise.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
                copy=capture_owns_inputs,
            ),
            num_steps=num_steps,
        )
        if not self.enable_cuda_graph or self.graph is None:
            actions = self._context_forward(infer_state)
        else:
            actions = self.graph.run(
                graph_key,
                infer_state,
                self._context_forward,
            )
        end_event.record()
        return Pi0ActionOutput(
            actions=actions[..., :action_dim],
            timing_events=(start_event, end_event),
        )
