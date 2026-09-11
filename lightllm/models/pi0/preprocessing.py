from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from lightllm.utils.config_utils import get_pi_config


# Benchmark camera names are an external I/O concern.  Resolve them once at
# startup into the checkpoint's canonical image keys; neither the HTTP action
# flow nor the inference hot path needs to know Pi's semantic camera roles.
_PI_DOMAIN_PROFILES = {
    "libero": {
        "state_dim": 8,
        "action_dim": 7,
        "image_map": {
            "agentview": "base",
            "observation/image": "base",
            "robot0_eye_in_hand": "left_wrist",
            "wrist": "left_wrist",
            "observation/wrist_image": "left_wrist",
        },
        "ignored_images": ["frontview", "galleryview"],
        "required_images": ["base", "left_wrist"],
    },
    "droid": {
        "state_dim": 8,
        "action_dim": 8,
        "image_map": {
            "observation/exterior_image_1_left": "base",
            "exterior_image_1_left": "base",
            "over_shoulder_left_camera": "base",
            "observation/wrist_image_left": "left_wrist",
            "wrist_image_left": "left_wrist",
            "wrist_cam": "left_wrist",
        },
        "ignored_images": ["over_shoulder_right_camera", "head_camera"],
        "required_images": ["base", "left_wrist"],
    },
    "aloha": {
        "state_dim": 14,
        "action_dim": 14,
        "robot_adapter": {"type": "openpi_aloha"},
        "image_map": {
            "cam_high": "base",
            "cam_left_wrist": "left_wrist",
            "cam_right_wrist": "right_wrist",
        },
        "ignored_images": ["cam_low"],
        "required_images": ["base"],
    },
    "robotwin": {
        "state_dim": 14,
        "action_dim": 14,
        "image_map": {
            "head_camera": "base",
            "left_camera": "left_wrist",
            "right_camera": "right_wrist",
        },
        "ignored_images": ["front_camera", "observer_camera"],
        "required_images": ["base", "left_wrist", "right_wrist"],
    },
}

_PI_DOMAIN_ALIASES = {
    "pi0_libero": "libero",
    "x2v_libero": "libero",
    "pi0_droid": "droid",
    "robolab": "droid",
    "x2v_robolab": "droid",
    "pi0_aloha": "aloha",
    "x2v_robotwin": "robotwin",
}

_PI_IMAGE_ROLE_SUFFIXES = {
    "base": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}

_PI_DEFAULT_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def _load_optional_json(path: str | None, label: str):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            value = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load {label} JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value


def _checkpoint_image_key(target: str, image_keys: tuple[str, ...]) -> str:
    if target in image_keys:
        return target
    suffix = _PI_IMAGE_ROLE_SUFFIXES.get(target, target)
    matches = [key for key in image_keys if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) != 1:
        raise ValueError(
            f"Pi checkpoint has no unambiguous image slot for {target!r}: "
            f"{list(image_keys)!r}"
        )
    return matches[0]


def build_pi_domain_profiles(
    image_keys: tuple[str, ...], overrides: dict | None = None
) -> dict[str, dict]:
    """Build Pi I/O profiles with only canonical checkpoint keys as outputs."""

    profiles = copy.deepcopy(_PI_DOMAIN_PROFILES)
    if overrides is not None:
        raw_profiles = overrides.get("domains", overrides)
        if not isinstance(raw_profiles, dict):
            raise ValueError("domain config must contain a domain object")
        for raw_name, raw_profile in raw_profiles.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("domain names must be non-empty strings")
            if not isinstance(raw_profile, dict):
                raise ValueError(f"domain profile {raw_name!r} must be an object")
            name = raw_name.strip().lower()
            profile = dict(profiles.get(name, {}))
            for key, value in raw_profile.items():
                if key == "image_map" and key in profile:
                    if not isinstance(value, dict):
                        raise ValueError(f"domain={name!r} image_map must be an object")
                    image_map = dict(profile[key])
                    image_map.update(value)
                    profile[key] = image_map
                else:
                    profile[key] = value
            profiles[name] = profile

    for name, profile in profiles.items():
        for key in ("state_dim", "action_dim"):
            value = profile.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"domain={name!r} requires a positive {key}")
        image_map = profile.get("image_map", {})
        if not isinstance(image_map, dict) or any(
            not isinstance(source, str) or not isinstance(target, str)
            for source, target in image_map.items()
        ):
            raise ValueError(f"domain={name!r} image_map must map strings to strings")
        profile["image_map"] = {
            source: _checkpoint_image_key(target, image_keys)
            for source, target in image_map.items()
        }
        for key in ("ignored_images", "required_images"):
            values = profile.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) for value in values
            ):
                raise ValueError(f"domain={name!r} {key} must be a string array")
        profile["required_images"] = [
            _checkpoint_image_key(target, image_keys)
            for target in profile.get("required_images", [])
        ]
    return profiles


def load_pi_action_config(args) -> dict:
    """Resolve checkpoint and benchmark I/O once before child processes start."""

    config = get_pi_config(
        args.model_dir,
        action_dim=args.vla_action_dim,
        action_horizon=args.vla_action_horizon,
        num_denoise_steps=args.vla_num_denoise_steps,
        robot_adapter=_load_optional_json(args.vla_robot_adapter, "robot adapter"),
        norm_config=_load_optional_json(args.vla_norm_config, "normalization config"),
        action_postprocess_config=_load_optional_json(
            args.vla_action_postprocess_config, "action postprocess config"
        ),
        dtype=args.data_type,
    )
    image_keys = tuple(config["image_keys"]) or _PI_DEFAULT_IMAGE_KEYS
    config["image_keys"] = image_keys
    domain_overrides = _load_optional_json(
        getattr(args, "vla_domain_config", None), "domain config"
    )
    config["domain_aliases"] = dict(_PI_DOMAIN_ALIASES)
    if domain_overrides is not None:
        aliases = domain_overrides.get("aliases", {})
        if not isinstance(aliases, dict) or any(
            not isinstance(alias, str) or not isinstance(domain, str)
            for alias, domain in aliases.items()
        ):
            raise ValueError("domain aliases must map strings to strings")
        config["domain_aliases"].update(
            {
                alias.strip().lower(): domain.strip().lower()
                for alias, domain in aliases.items()
            }
        )
    config["domain_profiles"] = build_pi_domain_profiles(image_keys, domain_overrides)
    for name, profile in config["domain_profiles"].items():
        if profile["state_dim"] > config["max_state_dim"]:
            raise ValueError(f"domain={name!r} state_dim exceeds checkpoint maximum")
        if profile["action_dim"] > config["max_action_dim"]:
            raise ValueError(f"domain={name!r} action_dim exceeds checkpoint maximum")
    return config


class NormalizationMode(str, Enum):
    IDENTITY = "identity"
    MEAN_STD = "mean_std"
    QUANTILES = "quantiles"


@dataclass
class NormalizationSpec:
    mode: NormalizationMode = NormalizationMode.IDENTITY
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    q01: Optional[torch.Tensor] = None
    q99: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "NormalizationSpec":
        if not value:
            return cls()
        mode = str(value.get("mode", value.get("type", "identity"))).lower()
        aliases = {"meanstd": "mean_std", "zscore": "mean_std", "quantile": "quantiles"}
        result = cls(mode=NormalizationMode(aliases.get(mode, mode)))
        for name in ("mean", "std", "q01", "q99"):
            if value.get(name) is not None:
                setattr(result, name, torch.as_tensor(value[name], dtype=torch.float32))
        if result.mode is NormalizationMode.MEAN_STD and (
            result.mean is None or result.std is None
        ):
            raise ValueError("mean_std normalization requires mean and std")
        if result.mode is NormalizationMode.QUANTILES and (
            result.q01 is None or result.q99 is None
        ):
            raise ValueError("quantile normalization requires q01 and q99")
        return result

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        dim = value.shape[-1]
        if self.mode is NormalizationMode.IDENTITY:
            return value
        if self.mode is NormalizationMode.MEAN_STD:
            mean = self.mean[..., :dim].to(value.device)
            std = self.std[..., :dim].to(value.device)
            return (value - mean) / (std + 1e-6)
        q01 = self.q01[..., :dim].to(value.device)
        q99 = self.q99[..., :dim].to(value.device)
        return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        dim = value.shape[-1]
        if self.mode is NormalizationMode.IDENTITY:
            return value
        if self.mode is NormalizationMode.MEAN_STD:
            stats_dim = self.mean.shape[-1]
            handled_dim = min(dim, stats_dim)
            # OpenPI pads missing means with 0 and stds with 1, then applies
            # the same epsilon to every dimension.
            output = value * (1.0 + 1e-6)
            mean = self.mean[..., :handled_dim].to(value.device)
            std = self.std[..., :handled_dim].to(value.device)
            output[..., :handled_dim] = value[..., :handled_dim] * (std + 1e-6) + mean
            return output
        stats_dim = self.q01.shape[-1]
        handled_dim = min(dim, stats_dim)
        output = value.clone()
        q01 = self.q01[..., :handled_dim].to(value.device)
        q99 = self.q99[..., :handled_dim].to(value.device)
        output[..., :handled_dim] = (value[..., :handled_dim] + 1.0) / 2.0 * (
            q99 - q01 + 1e-6
        ) + q01
        return output


class _DomainTransform:
    """Normalization and robot-coordinate conversion for one domain."""

    def __init__(self, config: dict):
        norm = config.get("norm_config") or {}
        state_config = norm.get("state", norm.get("observation.state"))
        action_config = norm.get("action", norm.get("actions"))
        self.state_normalizer = NormalizationSpec.from_dict(state_config)
        self.action_normalizer = NormalizationSpec.from_dict(action_config)
        self.robot_adapter = config.get("robot_adapter") or {}
        self.action_postprocess_config = config.get("action_postprocess_config") or {}

    def normalize_state(self, state) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32)
        if self.robot_adapter.get("type") == "openpi_aloha":
            # Official OpenPI ALOHA adapter: flip joint signs and convert the
            # two normalized linear grippers to Pi's angular coordinates.
            state = state.clone()
            flip = state.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
            state = state * flip
            linear = state[..., [6, 13]] * (0.05800 - 0.01844) + 0.01844
            ratio = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
            angular = torch.asin(ratio.clamp(-1.0, 1.0))
            state[..., [6, 13]] = (angular - 0.5476) / (1.6296 - 0.5476)
        return self.state_normalizer.normalize(state)

    def postprocess_actions(
        self, actions: torch.Tensor, raw_state=None
    ) -> torch.Tensor:
        result = self.action_normalizer.unnormalize(actions)
        if self.robot_adapter.get("type") == "openpi_aloha":
            # Convert Pi angular actions back to the standard ALOHA space.
            flip = result.new_tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
            result = result * flip
            grippers = result[..., [6, 13]] + 0.5476
            result[..., [6, 13]] = (grippers - (-0.6213)) / (1.4910 - (-0.6213))
        relative_mask = self.robot_adapter.get("relative_action_mask")
        if relative_mask is not None and raw_state is not None:
            mask = torch.as_tensor(
                relative_mask, dtype=torch.bool, device=result.device
            )
            state = torch.as_tensor(raw_state, dtype=result.dtype, device=result.device)
            dims = min(mask.numel(), result.shape[-1], state.shape[-1])
            if result.ndim == 2:
                state_row = state if state.ndim == 1 else state[0]
                offset = torch.where(
                    mask[:dims],
                    state_row[:dims],
                    torch.zeros_like(state_row[:dims]),
                )
            else:
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                if state.shape[0] != result.shape[0]:
                    raise ValueError("raw state batch does not match action batch")
                offset = torch.where(
                    mask[None, :dims],
                    state[:, :dims],
                    torch.zeros_like(state[:, :dims]),
                )[:, None, :]
            result[..., :dims] += offset
        if self.action_postprocess_config.get("scale") is not None:
            result = result * torch.as_tensor(
                self.action_postprocess_config["scale"],
                dtype=result.dtype,
                device=result.device,
            )
        if self.action_postprocess_config.get("offset") is not None:
            result = result + torch.as_tensor(
                self.action_postprocess_config["offset"],
                dtype=result.dtype,
                device=result.device,
            )
        if self.action_postprocess_config.get("clip") is not None:
            lower, upper = self.action_postprocess_config["clip"]
            result = result.clamp(lower, upper)
        return result


class Pi0PrePostProcessor:
    """Select OpenPI-compatible transforms by environment domain.

    Pi itself has no learned domain id.  A domain only selects the coordinate
    system and normalization statistics wrapped around the fixed Pi model.
    """

    def __init__(self, config: dict):
        self.default = _DomainTransform(config)
        self.domains = {}
        for name, profile in (config.get("domain_profiles") or {}).items():
            domain_config = dict(config)
            for key in (
                "norm_config",
                "robot_adapter",
                "action_postprocess_config",
            ):
                if key in profile:
                    domain_config[key] = profile[key]
            self.domains[name] = _DomainTransform(domain_config)

    def _select(self, domain: str | None) -> _DomainTransform:
        if domain is None:
            return self.default
        try:
            return self.domains[domain]
        except KeyError as exc:
            raise ValueError(f"unknown Pi action domain: {domain!r}") from exc

    def normalize_state(self, state, *, domain: str | None = None) -> torch.Tensor:
        return self._select(domain).normalize_state(state)

    def postprocess_actions(
        self,
        actions: torch.Tensor,
        raw_state=None,
        *,
        domain: str | None = None,
    ) -> torch.Tensor:
        return self._select(domain).postprocess_actions(actions, raw_state=raw_state)
