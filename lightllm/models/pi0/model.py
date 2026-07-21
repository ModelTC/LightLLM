from __future__ import annotations

from dataclasses import dataclass

import torch

from lightllm.common.basemodel.attention import get_prefill_att_backend_class
from lightllm.common.quantization import Quantcfg
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import (
    Gemma_2bTransformerLayerInfer,
)
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.models.pi0.infer_struct import Pi0ActionInferStateInfo
from lightllm.models.pi0.layer_infer import (
    Pi0ActionPostLayerInfer,
    Pi0ActionPreLayerInfer,
    Pi0ActionTransformerLayerInfer,
    Pi0VLMPreLayerInfer,
)
from lightllm.models.pi0.layer_weights import (
    Pi0ActionPreAndPostLayerWeight,
    Pi0ActionTransformerLayerWeight,
    Pi0SafeTensorLoader,
    Pi0VLMPreAndPostLayerWeight,
    Pi0VLMTransformerLayerWeight,
)
from lightllm.models.pi0.math_utils import denoise_schedule
from lightllm.models.registry import ModelRegistry
from lightllm.models.pi0.model_io import VLAActionModelOutput


def _stream_matching_weights(loader, target, predicate) -> None:
    """Feed matching tensors through the normal BaseWeight loading API."""

    for name in loader.keys():
        if not predicate(name):
            continue
        value = loader.tensor(name, device="cpu")
        target.load_hf_weights({name: value})
        del value


@ModelRegistry(["pi0", "pi05"], is_multimodal=True)
class Pi0VLMModel(Gemma_2bTpPartModel):
    """PaliGemma prefix model implemented by the normal BaseModel pipeline."""

    pre_and_post_weight_class = Pi0VLMPreAndPostLayerWeight
    transformer_weight_class = Pi0VLMTransformerLayerWeight
    pre_layer_infer_class = Pi0VLMPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Gemma_2bTransformerLayerInfer

    # The only attention-policy difference from an ordinary Gemma prefill.
    prefill_causal = False

    def __init__(self, kvargs: dict):
        self.vla_config: Pi0VLAConfig = kvargs.get("vla_config")
        if self.vla_config is None:
            self.vla_config = Pi0VLAConfig.from_model_dir(kvargs["weight_dir"], dtype=kvargs.get("data_type"))
        self.vla_config = self.vla_config.validate()
        super().__init__(kvargs)

    def _init_config(self):
        config = self.vla_config
        self.config = {
            "model_type": "gemma",
            "hidden_size": config.vlm_hidden_size,
            "n_embed": config.vlm_hidden_size,
            "num_hidden_layers": config.depth,
            "n_layer": config.depth,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.vlm_intermediate_size,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": self.max_seq_length,
            "tie_word_embeddings": True,
        }

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_

    def _init_custom(self):
        # The action expert consumes the owner's standard request table both
        # through CUDA IPC and in single-process parity tests.
        self.mem_manager.req_to_token_indexs = self.req_manager.req_to_token_indexs
        self.use_ieee_fp32_attention = self.data_type is torch.float32
        super()._init_custom()

    def _load_hf_weights(self):
        with Pi0SafeTensorLoader(self.vla_config.checkpoint_path) as loader:
            prepost_names = {
                "paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.paligemma.model.language_model.norm.weight",
            }
            _stream_matching_weights(loader, self.pre_post_weight, lambda name: name in prepost_names)
            for layer_index, layer_weight in enumerate(self.trans_layers_weight):
                prefix = "paligemma_with_expert.paligemma.model.language_model.layers." f"{layer_index}."
                _stream_matching_weights(
                    loader,
                    layer_weight,
                    lambda name, prefix=prefix: name.startswith(prefix),
                )
        self.pre_post_weight.verify_load()
        for layer_weight in self.trans_layers_weight:
            layer_weight.verify_load()


@dataclass
class _Pi0ActionRuntime:
    state: torch.Tensor | None
    noise: torch.Tensor
    action_horizon: int
    action_infer_state: Pi0ActionInferStateInfo
    timesteps: torch.Tensor
    dt: torch.Tensor


class Pi0ActionExpertModel:
    """Independent diffusion controller using standard LightLLM components."""

    def __init__(
        self,
        config: Pi0VLAConfig,
        mem_manager,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
        tp_group=None,
        quant_type: str = "none",
        quant_cfg_path: str | None = None,
    ):
        self.vla_config = config.validate()
        self.device = torch.device(device)
        self.data_type = dtype or config.torch_dtype
        self.mem_manager = mem_manager
        self.tp_group = tp_group
        self.layers_num = config.depth
        self.max_seq_length = mem_manager.req_to_token_indexs.shape[1]
        self.graph_max_batch_size = 32
        self.graph_max_len_in_batch = self.max_seq_length
        self.config = {
            "model_type": config.model_type.value,
            "hidden_size": config.expert_hidden_size,
            "n_embed": config.expert_hidden_size,
            "num_hidden_layers": config.depth,
            "n_layer": config.depth,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "intermediate_size": config.expert_intermediate_size,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": 1e-6,
            "is_pi05": config.is_pi05,
            "max_action_dim": config.max_action_dim,
            "max_state_dim": config.max_state_dim,
            "min_period": config.min_period,
            "max_period": config.max_period,
        }
        self.quant_cfg = Quantcfg(
            self.config,
            quant_type=quant_type or "none",
            custom_cfg_path=quant_cfg_path,
        )
        self.prefill_att_backend = get_prefill_att_backend_class(index=0)(model=self)
        self.pre_infer = Pi0ActionPreLayerInfer(self.config)
        self.post_infer = Pi0ActionPostLayerInfer(self.config)
        self.pre_post_weight = Pi0ActionPreAndPostLayerWeight(self.data_type, self.config, self.quant_cfg)
        self.trans_layers_weight = [
            Pi0ActionTransformerLayerWeight(
                layer_index,
                self.data_type,
                self.config,
                self.quant_cfg,
            )
            for layer_index in range(config.depth)
        ]
        self.layers_infer = [
            Pi0ActionTransformerLayerInfer(layer_index, self.config) for layer_index in range(config.depth)
        ]
        self._load_weights()
        self._init_rope_cache()

    def _load_weights(self):
        with Pi0SafeTensorLoader(self.vla_config.checkpoint_path) as loader:
            action_names = {
                name
                for name in loader.keys()
                if name.startswith(
                    (
                        "action_in_proj.",
                        "action_out_proj.",
                        "action_time_mlp_",
                        "time_mlp_",
                        "state_proj.",
                        "paligemma_with_expert.gemma_expert.model.norm.",
                    )
                )
            }
            _stream_matching_weights(loader, self.pre_post_weight, lambda name: name in action_names)
            for layer_index, layer_weight in enumerate(self.trans_layers_weight):
                prefix = "paligemma_with_expert.gemma_expert.model.layers." f"{layer_index}."
                _stream_matching_weights(
                    loader,
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
                self.vla_config.head_dim,
                2,
                dtype=torch.float32,
                device=self.device,
            )
            / self.vla_config.head_dim
        )
        inverse_frequency = 1.0 / (10000.0**exponents)
        radians = torch.outer(positions, inverse_frequency)
        self._cos_cached = radians.cos().to(self.data_type)
        self._sin_cached = radians.sin().to(self.data_type)

    def _create_action_infer_state(
        self,
        *,
        req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        query_length: int,
        ready_offset: int,
        sequence_offset: int,
    ) -> Pi0ActionInferStateInfo:
        return Pi0ActionInferStateInfo.create(
            self,
            req_indexes=req_indexes,
            prefix_seq_lens=prefix_seq_lens,
            query_length=query_length,
            ready_offset=ready_offset,
            sequence_offset=sequence_offset,
        )

    @staticmethod
    def _pad_last_dim(value: torch.Tensor, target: int) -> torch.Tensor:
        if value.shape[-1] > target:
            raise ValueError(f"input dimension {value.shape[-1]} exceeds checkpoint dimension {target}")
        if value.shape[-1] == target:
            return value
        return torch.nn.functional.pad(value, (0, target - value.shape[-1]))

    def _validate_action_metadata(
        self,
        *,
        prefix_req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        scratch_mem_indexes: torch.Tensor,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        action_horizon: int,
    ) -> None:
        if noise.ndim != 3 or noise.shape[1] != action_horizon:
            raise ValueError("noise must have shape [batch, action_horizon, action_dim]")
        batch_size = noise.shape[0]
        if prefix_req_indexes.shape != (batch_size,):
            raise ValueError("one prefix request index is required per action row")
        if prefix_seq_lens.shape != (batch_size,):
            raise ValueError("one prefix sequence length is required per action row")
        suffix_length = action_horizon + (0 if self.vla_config.is_pi05 else 1)
        if scratch_mem_indexes.numel() != batch_size * suffix_length:
            raise ValueError("scratch KV allocation does not match action suffix")
        if state is not None and (state.ndim != 2 or state.shape[0] != batch_size):
            raise ValueError("state must have shape [batch, state_dim]")
        if prefix_seq_lens.device.type == "cpu":
            prefix_lengths = prefix_seq_lens.tolist()
            if any(length <= 0 for length in prefix_lengths):
                raise ValueError("action prefix length must be positive")
            if any(length + suffix_length > self.max_seq_length for length in prefix_lengths):
                raise ValueError("action suffix exceeds the request table")
        return

    def _scratch_indexes_on_device(self, scratch_mem_indexes: torch.Tensor, expected_numel: int) -> torch.Tensor:
        get_lease = getattr(self.mem_manager, "get_scratch_write_indexes", None)
        if get_lease is not None:
            return get_lease(expected_numel)
        return scratch_mem_indexes.to(device=self.device, dtype=torch.int32, non_blocking=True).reshape(-1)

    def _create_action_runtime(
        self,
        *,
        prefix_req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        scratch_mem_indexes: torch.Tensor,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        num_steps: int,
    ) -> _Pi0ActionRuntime:
        _, action_horizon = noise.shape[:2]
        suffix_length = action_horizon + (0 if self.vla_config.is_pi05 else 1)
        state_infer_state = None
        if self.vla_config.is_pi05:
            ready_offset = 0
        else:
            state_infer_state = self._create_action_infer_state(
                req_indexes=prefix_req_indexes,
                prefix_seq_lens=prefix_seq_lens,
                query_length=1,
                ready_offset=0,
                sequence_offset=1,
            )
            ready_offset = 1
        action_infer_state = self._create_action_infer_state(
            req_indexes=prefix_req_indexes,
            prefix_seq_lens=prefix_seq_lens,
            query_length=action_horizon,
            ready_offset=ready_offset,
            sequence_offset=suffix_length,
        )
        position_offsets = torch.arange(suffix_length, dtype=torch.int32, device=self.device)
        position_ids = prefix_seq_lens[:, None] + position_offsets[None, :]
        flat_positions = position_ids.reshape(-1).long()
        position_cos = torch.index_select(self._cos_cached, 0, flat_positions)
        position_sin = torch.index_select(self._sin_cached, 0, flat_positions)
        action_infer_state.mem_index = scratch_mem_indexes
        action_infer_state.position_cos = position_cos
        action_infer_state.position_sin = position_sin
        action_infer_state.state_infer_state = state_infer_state
        timesteps, dt = denoise_schedule(num_steps, device=self.device)
        runtime = _Pi0ActionRuntime(
            state=state,
            noise=noise,
            action_horizon=action_horizon,
            action_infer_state=action_infer_state,
            timesteps=timesteps,
            dt=dt,
        )
        return runtime

    def _denoise_step_runtime(
        self,
        runtime: _Pi0ActionRuntime,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_dim: int,
    ) -> torch.Tensor:
        hidden_states, condition = self.pre_infer.context_forward(
            runtime.state, noisy_actions, timestep, self.pre_post_weight
        )
        runtime.action_infer_state.condition = condition
        for layer, weight in zip(self.layers_infer, self.trans_layers_weight, strict=True):
            hidden_states = layer.context_forward(
                hidden_states,
                runtime.action_infer_state,
                weight,
            )
        return self.post_infer.token_forward(
            hidden_states,
            condition,
            runtime.action_horizon,
            action_dim,
            self.pre_post_weight,
        )

    def _sample_actions_runtime(self, runtime: _Pi0ActionRuntime) -> torch.Tensor:
        actions = runtime.noise
        for timestep in runtime.timesteps:
            velocity = self._denoise_step_runtime(
                runtime,
                actions,
                timestep.expand(actions.shape[0]),
                action_dim=self.vla_config.max_action_dim,
            )
            actions = actions + runtime.dt * velocity
        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        prefix_req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        scratch_mem_indexes: torch.Tensor,
        num_steps: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
        prefix_version: int = 0,
    ) -> VLAActionModelOutput:
        num_steps = num_steps or self.vla_config.num_denoise_steps
        action_dim = action_dim or noise.shape[-1]
        action_horizon = action_horizon or noise.shape[-2]
        if action_dim > self.vla_config.max_action_dim:
            raise ValueError("requested action_dim exceeds checkpoint maximum")
        if prefix_seq_lens.shape[0] != noise.shape[0]:
            raise ValueError("prefix and noise batch sizes differ")
        if not self.vla_config.is_pi05 and state is None:
            raise ValueError("pi0 action sampling requires continuous state")
        noise = self._pad_last_dim(noise, self.vla_config.max_action_dim)
        state = None if state is None else self._pad_last_dim(state, self.vla_config.max_state_dim)
        self._validate_action_metadata(
            prefix_req_indexes=prefix_req_indexes,
            prefix_seq_lens=prefix_seq_lens,
            scratch_mem_indexes=scratch_mem_indexes,
            state=state,
            noise=noise,
            action_horizon=action_horizon,
        )
        expected_scratch = noise.shape[0] * (action_horizon + (0 if self.vla_config.is_pi05 else 1))
        scratch_device = self._scratch_indexes_on_device(scratch_mem_indexes, expected_scratch)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        runtime = self._create_action_runtime(
            prefix_req_indexes=prefix_req_indexes.to(self.device, dtype=torch.int32, non_blocking=True),
            prefix_seq_lens=prefix_seq_lens.to(self.device, dtype=torch.int32, non_blocking=True),
            scratch_mem_indexes=scratch_device,
            state=None if state is None else state.to(self.device, dtype=torch.float32, non_blocking=True),
            noise=noise.to(self.device, dtype=torch.float32, non_blocking=True),
            num_steps=num_steps,
        )
        actions = self._sample_actions_runtime(runtime)
        end_event.record()
        return VLAActionModelOutput(
            logits=None,
            actions=actions[..., :action_dim],
            action_status="HAS_OUTPUT",
            prefix_version=prefix_version,
            safe_to_release=True,
            policy_timing={},
            action_timing_events=(start_event, end_event),
        )
