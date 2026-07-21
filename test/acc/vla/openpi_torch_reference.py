"""Independent PyTorch execution of the official OpenPI pi0 equations.

This oracle intentionally uses Transformers' Gemma attention/MLP modules and
DynamicCache rather than LightLLM's pi0 layer inference implementation. It is
kept test-only and loads the same converted checkpoint component by component.
"""

from __future__ import annotations

import math

import numpy as np
import sentencepiece
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GemmaConfig, SiglipVisionConfig, SiglipVisionModel
from transformers.cache_utils import DynamicCache
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRotaryEmbedding,
)

from lightllm.models.pi0.config import Pi0VLAConfig
from lightllm.models.pi0.layer_weights.loader import Pi0SafeTensorLoader


OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def _reference_time_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    compute_dtype = torch.float32 if time.device.type == "mps" else torch.float64
    fraction = torch.linspace(
        0.0,
        1.0,
        dimension // 2,
        dtype=compute_dtype,
        device=time.device,
    )
    period = min_period * (max_period / min_period) ** fraction
    radians = (2.0 * math.pi / period)[None, :] * time.to(compute_dtype)[:, None]
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)


def _reference_denoise_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    times = 1.0 + torch.arange(num_steps, dtype=torch.float32, device=device) * dt
    return times, dt


def reference_tokenize(
    config: Pi0VLAConfig,
    prompts: list[str],
    states: torch.Tensor,
    tokenizer_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent implementation of OpenPI's PaligemmaTokenizer."""
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
    rows = []
    masks = []
    state_values = states.detach().cpu().numpy()
    for index, prompt in enumerate(prompts):
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if config.is_pi05:
            bins = np.digitize(state_values[index], bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_text = " ".join(map(str, bins))
            text = f"Task: {cleaned}, State: {state_text};\nAction: "
            tokens = tokenizer.encode(text, add_bos=True)
        else:
            tokens = tokenizer.encode(cleaned, add_bos=True) + tokenizer.encode("\n")
        tokens = tokens[: config.tokenizer_max_length]
        mask = [True] * len(tokens)
        padding = config.tokenizer_max_length - len(tokens)
        rows.append(tokens + [0] * padding)
        masks.append(mask + [False] * padding)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _reference_preprocess_image(images: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """Independent copy of OpenPI's inference image transform."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    if images.dtype == torch.uint8:
        images = images.float() / 255.0 * 2.0 - 1.0
    else:
        images = images.float()
    _, _, current_height, current_width = images.shape
    target_height, target_width = resolution
    if (current_height, current_width) == resolution:
        return images
    ratio = max(current_width / target_width, current_height / target_height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).clamp(-1.0, 1.0)
    pad_top, extra_height = divmod(target_height - resized_height, 2)
    pad_left, extra_width = divmod(target_width - resized_width, 2)
    return F.pad(
        images,
        (
            pad_left,
            pad_left + extra_width,
            pad_top,
            pad_top + extra_height,
        ),
        value=-1.0,
    )


@torch.no_grad()
def reference_vision_encode(
    config: Pi0VLAConfig,
    images: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """OpenPI vision oracle independent of LightLLM's visual component."""
    device = torch.device(device)
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=14,
        image_size=config.image_resolution[0],
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        projection_dim=config.vlm_hidden_size,
        vision_use_head=False,
    )
    vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        vision_model = SiglipVisionModel(vision_config)
    checkpoint_prefix = "paligemma_with_expert.paligemma.model.vision_tower."
    with Pi0SafeTensorLoader(config.checkpoint_path) as loader:
        state = {
            key[len(checkpoint_prefix) :]: loader.tensor(key, device=device, dtype=torch.float32)
            for key in loader.keys()
            if key.startswith(checkpoint_prefix)
        }
        missing, unexpected = vision_model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"reference vision load mismatch: missing={missing}, unexpected={unexpected}")
        embeddings = vision_model.vision_model.embeddings
        embeddings.position_ids = torch.arange(embeddings.num_patches, dtype=torch.long, device=device).expand(1, -1)
        projector_weight = loader.tensor(
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
            device=device,
            dtype=torch.float32,
        )
        projector_bias = loader.tensor(
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
            device=device,
            dtype=torch.float32,
        )
    vision_model.eval()
    pixels = _reference_preprocess_image(images, config.image_resolution).to(device)
    hidden_states = vision_model(pixel_values=pixels).last_hidden_state
    return F.linear(hidden_states, projector_weight, projector_bias)


def _additive_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(
        mask[:, None],
        torch.tensor(0.0, device=mask.device),
        torch.tensor(OPENPI_ATTENTION_MASK_VALUE, device=mask.device),
    )


def _make_mask(pad_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(ar_mask, dim=1)
    return (cumulative[:, None, :] <= cumulative[:, :, None]) & (pad_mask[:, None, :] & pad_mask[:, :, None])


class _ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, adaptive: bool):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.dense = nn.Linear(hidden_size, hidden_size * 3)
        else:
            self.weight = nn.Parameter(torch.empty(hidden_size))

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor | None):
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + 1e-6)
        if not self.adaptive:
            return (normalized.float() * (1.0 + self.weight.float())).to(hidden_states.dtype), None
        modulation = self.dense(condition).unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normalized = normalized.float() * (1.0 + scale.float()) + shift.float()
        return normalized.to(hidden_states.dtype), gate.to(hidden_states.dtype)


class _ReferenceDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_index: int, adaptive: bool):
        super().__init__()
        self.self_attn = GemmaAttention(config, layer_index)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = _ReferenceRMSNorm(config.hidden_size, adaptive)
        self.post_attention_layernorm = _ReferenceRMSNorm(config.hidden_size, adaptive)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DynamicCache,
        cache_position: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, condition)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states if gate is None else residual + hidden_states * gate
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, condition)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states if gate is None else residual + hidden_states * gate


class _ReferenceGemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig, adaptive: bool):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_ReferenceDecoderLayer(config, index, adaptive) for index in range(config.num_hidden_layers)]
        )
        self.norm = _ReferenceRMSNorm(config.hidden_size, adaptive)
        self.rotary_emb = GemmaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DynamicCache,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        start = cache.get_seq_length()
        cache_position = torch.arange(start, start + hidden_states.shape[1], device=hidden_states.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                cache,
                cache_position,
                condition,
            )
        hidden_states, _ = self.norm(hidden_states, condition)
        return hidden_states, cache


def _gemma_config(config: Pi0VLAConfig, component: str) -> GemmaConfig:
    hidden_size = config.vlm_hidden_size if component == "vlm" else config.expert_hidden_size
    intermediate = config.vlm_intermediate_size if component == "vlm" else config.expert_intermediate_size
    result = GemmaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        num_hidden_layers=config.depth,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_activation="gelu_pytorch_tanh",
        vocab_size=config.vocab_size,
    )
    result._attn_implementation = "eager"
    return result


def _load_gemma(
    loader: Pi0SafeTensorLoader,
    config: Pi0VLAConfig,
    component: str,
    device: torch.device,
) -> _ReferenceGemmaModel:
    adaptive = config.is_pi05 and component == "expert"
    gemma_config = _gemma_config(config, component)
    with torch.device("meta"):
        model = _ReferenceGemmaModel(gemma_config, adaptive)
    root = (
        "paligemma_with_expert.paligemma.model.language_model."
        if component == "vlm"
        else "paligemma_with_expert.gemma_expert.model."
    )
    state = {
        key[len(root) :]: loader.tensor(key, device=device, dtype=torch.float32)
        for key in loader.keys()
        if key.startswith(root) and ".embed_tokens." not in key
    }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"reference {component} load mismatch: missing={missing}, unexpected={unexpected}")
    # inv_freq is a non-persistent buffer, so assign=True cannot materialize the
    # meta instance from checkpoint state.
    model.rotary_emb = GemmaRotaryEmbedding(gemma_config, device=device)
    model.eval()
    return model


def _clone_cache(cache: DynamicCache, config: GemmaConfig) -> DynamicCache:
    legacy = cache.to_legacy_cache()
    return DynamicCache(((key.clone(), value.clone()) for key, value in legacy), config=config)


class OpenPiTorchReference:
    def __init__(self, config: Pi0VLAConfig, *, device: torch.device | str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        with Pi0SafeTensorLoader(config.checkpoint_path) as loader:
            self.vlm = _load_gemma(loader, config, "vlm", self.device)
            self.expert = _load_gemma(loader, config, "expert", self.device)
            self.token_embedding = loader.tensor(
                "paligemma_with_expert.paligemma.lm_head.weight",
                device=self.device,
                dtype=torch.float32,
            )
            names = {
                "action_in_weight": "action_in_proj.weight",
                "action_in_bias": "action_in_proj.bias",
                "action_out_weight": "action_out_proj.weight",
                "action_out_bias": "action_out_proj.bias",
            }
            if config.is_pi05:
                names.update(
                    {
                        "time_in_weight": "time_mlp_in.weight",
                        "time_in_bias": "time_mlp_in.bias",
                        "time_out_weight": "time_mlp_out.weight",
                        "time_out_bias": "time_mlp_out.bias",
                    }
                )
            else:
                names.update(
                    {
                        "state_weight": "state_proj.weight",
                        "state_bias": "state_proj.bias",
                        "time_in_weight": "action_time_mlp_in.weight",
                        "time_in_bias": "action_time_mlp_in.bias",
                        "time_out_weight": "action_time_mlp_out.weight",
                        "time_out_bias": "action_time_mlp_out.bias",
                    }
                )
            for attribute, key in names.items():
                setattr(
                    self,
                    attribute,
                    loader.tensor(key, device=self.device, dtype=torch.float32),
                )

    def prefill(
        self,
        image_embeds: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[DynamicCache, torch.Tensor]:
        embeddings = [value.to(self.device).float() for value in image_embeds]
        masks = [
            mask.to(self.device)[:, None].expand(embedding.shape[:2]).bool()
            for embedding, mask in zip(embeddings, image_masks, strict=True)
        ]
        language = F.embedding(tokens.to(self.device), self.token_embedding) * math.sqrt(self.config.vlm_hidden_size)
        embeddings.append(language)
        masks.append(token_mask.to(self.device).bool())
        hidden_states = torch.cat(embeddings, dim=1)
        pad_mask = torch.cat(masks, dim=1)
        ar_mask = torch.zeros_like(pad_mask)
        attention_mask = _additive_mask(_make_mask(pad_mask, ar_mask))
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        cache = DynamicCache(config=self.vlm.config)
        _, cache = self.vlm(hidden_states, attention_mask, position_ids, cache)
        return cache, pad_mask

    def _embed_suffix(
        self,
        state: torch.Tensor | None,
        actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        action_tokens = F.linear(actions, self.action_in_weight, self.action_in_bias)
        time_embedding = _reference_time_embedding(
            timestep,
            self.config.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
        ).float()
        if self.config.is_pi05:
            condition = F.silu(F.linear(time_embedding, self.time_in_weight, self.time_in_bias))
            condition = F.silu(F.linear(condition, self.time_out_weight, self.time_out_bias))
            hidden_states = action_tokens
            ar_mask = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.device)
            ar_mask[:, 0] = True
        else:
            state_token = F.linear(state, self.state_weight, self.state_bias)[:, None]
            time_tokens = time_embedding[:, None].expand_as(action_tokens)
            action_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_tokens = F.silu(F.linear(action_tokens, self.time_in_weight, self.time_in_bias))
            action_tokens = F.linear(action_tokens, self.time_out_weight, self.time_out_bias)
            hidden_states = torch.cat([state_token, action_tokens], dim=1)
            ar_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=self.device)
            ar_mask[:, :2] = True
            condition = None
        return hidden_states, ar_mask, condition

    @torch.no_grad()
    def sample_actions(
        self,
        cache: DynamicCache,
        prefix_pad_mask: torch.Tensor,
        state: torch.Tensor | None,
        noise: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        actions = noise.to(self.device).float()
        if state is not None:
            state = state.to(self.device).float()
        times, dt = _reference_denoise_schedule(num_steps, self.device)
        for scalar_time in times:
            timestep = scalar_time.expand(actions.shape[0])
            hidden_states, ar_mask, condition = self._embed_suffix(state, actions, timestep)
            suffix_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=self.device)
            suffix_attention = _make_mask(suffix_mask, ar_mask)
            prefix_attention = prefix_pad_mask[:, None].expand(
                hidden_states.shape[0], hidden_states.shape[1], prefix_pad_mask.shape[1]
            )
            attention_mask = _additive_mask(torch.cat([prefix_attention, suffix_attention], dim=-1))
            position_ids = prefix_pad_mask.sum(-1)[:, None] + torch.cumsum(suffix_mask, dim=1) - 1
            scratch_cache = _clone_cache(cache, self.expert.config)
            hidden_states, _ = self.expert(
                hidden_states,
                attention_mask,
                position_ids,
                scratch_cache,
                condition,
            )
            velocity = F.linear(
                hidden_states[:, -actions.shape[1] :].float(),
                self.action_out_weight,
                self.action_out_bias,
            )
            actions = actions + dt * velocity
        return actions
