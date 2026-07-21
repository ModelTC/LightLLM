import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch


class VLAModelType(str, Enum):
    PI0 = "pi0"
    PI05 = "pi05"


class StateMode(str, Enum):
    SUFFIX_CONTINUOUS = "suffix_continuous"
    PREFIX_DISCRETE = "prefix_discrete"


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}

_VLM_LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


def read_checkpoint_vocab_size(checkpoint_path: str | Path) -> int:
    """Read the PaliGemma vocabulary width without loading its weights."""

    from safetensors import safe_open

    try:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            if _VLM_LM_HEAD_KEY not in checkpoint.keys():
                raise ValueError(f"VLA checkpoint is missing {_VLM_LM_HEAD_KEY}")
            shape = checkpoint.get_slice(_VLM_LM_HEAD_KEY).get_shape()
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read VLA vocabulary size from {checkpoint_path}") from exc
    if len(shape) != 2 or int(shape[0]) <= 0:
        raise ValueError(f"invalid VLA lm-head shape for {_VLM_LM_HEAD_KEY}: {shape}")
    return int(shape[0])


def parse_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return _DTYPES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported VLA dtype: {value}") from exc


@dataclass(frozen=True)
class Pi0VLAConfig:
    """Runtime configuration shared by the pi0 and pi0.5 model processes.

    The supplied checkpoints are LeRobot exports of OpenPI models. This object
    normalizes their policy config into the fields needed by LightLLM without
    rewriting the checkpoint on disk.
    """

    model_type: VLAModelType
    model_dir: str
    vocab_size: int
    action_dim: int = 32
    action_horizon: int = 50
    num_denoise_steps: int = 10
    state_mode: StateMode = StateMode.SUFFIX_CONTINUOUS
    robot_adapter: Optional[dict[str, Any]] = None
    norm_config: Optional[dict[str, Any]] = None
    action_postprocess_config: Optional[dict[str, Any]] = None
    dtype: str = "float32"
    image_resolution: tuple[int, int] = (224, 224)
    image_keys: tuple[str, ...] = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    tokenizer_max_length: int = 48
    min_period: float = 0.004
    max_period: float = 4.0
    max_action_dim: int = 32
    max_state_dim: int = 32
    depth: int = 18
    vlm_hidden_size: int = 2048
    expert_hidden_size: int = 1024
    vlm_intermediate_size: int = 16384
    expert_intermediate_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256

    @property
    def torch_dtype(self) -> torch.dtype:
        return parse_torch_dtype(self.dtype)

    @property
    def is_pi05(self) -> bool:
        return self.model_type is VLAModelType.PI05

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.model_dir) / "model.safetensors")

    def with_overrides(
        self,
        *,
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        num_denoise_steps: Optional[int] = None,
    ) -> "Pi0VLAConfig":
        return replace(
            self,
            action_dim=self.action_dim if action_dim is None else action_dim,
            action_horizon=self.action_horizon if action_horizon is None else action_horizon,
            num_denoise_steps=self.num_denoise_steps if num_denoise_steps is None else num_denoise_steps,
        ).validate()

    def validate(self) -> "Pi0VLAConfig":
        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            raise ValueError(f"VLA model_dir does not exist: {model_dir}")
        if not Path(self.checkpoint_path).is_file():
            raise ValueError(f"VLA checkpoint does not exist: {self.checkpoint_path}")
        if not 0 < self.action_dim <= self.max_action_dim:
            raise ValueError(f"action_dim must be in [1, {self.max_action_dim}], got {self.action_dim}")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        if any(value <= 0 for value in self.image_resolution):
            raise ValueError("image_resolution values must be positive")
        if self.tokenizer_max_length <= 0:
            raise ValueError("tokenizer_max_length must be positive")
        if self.model_type is VLAModelType.PI0 and self.state_mode is not StateMode.SUFFIX_CONTINUOUS:
            raise ValueError("pi0 requires state_mode=suffix_continuous")
        if self.model_type is VLAModelType.PI05 and self.state_mode is not StateMode.PREFIX_DISCRETE:
            raise ValueError("pi0.5 requires state_mode=prefix_discrete")
        parse_torch_dtype(self.dtype)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        return self

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        *,
        model_type: str | VLAModelType | None = None,
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        num_denoise_steps: Optional[int] = None,
        state_mode: str | StateMode | None = None,
        robot_adapter: Optional[dict[str, Any]] = None,
        norm_config: Optional[dict[str, Any]] = None,
        action_postprocess_config: Optional[dict[str, Any]] = None,
        dtype: str | torch.dtype | None = None,
    ) -> "Pi0VLAConfig":
        path = Path(model_dir)
        try:
            raw = json.loads((path / "config.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read VLA config from {path / 'config.json'}") from exc

        checkpoint_type = raw.get("model_type") or raw.get("type")
        resolved_type = VLAModelType(model_type or checkpoint_type)
        if model_type is not None and checkpoint_type is not None:
            if resolved_type is not VLAModelType(checkpoint_type):
                raise ValueError(
                    f"requested model_type={resolved_type.value} does not match "
                    f"checkpoint model_type={checkpoint_type}"
                )
        expected_state_mode = (
            StateMode.PREFIX_DISCRETE if resolved_type is VLAModelType.PI05 else StateMode.SUFFIX_CONTINUOUS
        )
        output_shape = raw.get("output_features", {}).get("action", {}).get("shape", [32])
        image_keys = tuple(
            key
            for key, feature in raw.get("input_features", {}).items()
            if str(feature.get("type", "")).upper() == "VISUAL"
        )
        config = cls(
            model_type=resolved_type,
            model_dir=str(path),
            # LeRobot policy config does not carry the PaliGemma vocabulary
            # width. The tied VLM lm-head tensor is the checkpoint authority.
            vocab_size=read_checkpoint_vocab_size(path / "model.safetensors"),
            action_dim=int(output_shape[-1]) if action_dim is None else action_dim,
            action_horizon=(
                int(raw.get("chunk_size", raw.get("action_horizon", 50))) if action_horizon is None else action_horizon
            ),
            num_denoise_steps=(
                int(raw.get("num_inference_steps", 10)) if num_denoise_steps is None else num_denoise_steps
            ),
            state_mode=StateMode(state_mode or expected_state_mode),
            robot_adapter=robot_adapter,
            norm_config=norm_config,
            action_postprocess_config=action_postprocess_config,
            dtype=(str(raw.get("dtype", "float32")) if dtype is None else str(dtype).removeprefix("torch.")),
            image_resolution=tuple(raw.get("image_resolution", (224, 224))),
            image_keys=image_keys
            or (
                "observation.images.base_0_rgb",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            ),
            tokenizer_max_length=int(
                raw.get(
                    "tokenizer_max_length",
                    200 if resolved_type is VLAModelType.PI05 else 48,
                )
            ),
            min_period=float(raw.get("min_period", 0.004)),
            max_period=float(raw.get("max_period", 4.0)),
            max_action_dim=int(raw.get("max_action_dim", 32)),
            max_state_dim=int(raw.get("max_state_dim", 32)),
        )
        return config.validate()

    @classmethod
    def from_start_args(cls, args) -> "Pi0VLAConfig":
        """Build the policy config from the ordinary LightLLM start args.

        Keeping this adapter next to the model config lets the router,
        actionserver, tokenizer, and HTTP workers resolve one identical view
        without introducing a second VLA server configuration layer.
        """

        def load_optional_json(path: Optional[str], label: str):
            if path is None:
                return None
            try:
                value = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"unable to load {label} JSON from {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label} JSON must contain an object")
            return value

        return cls.from_model_dir(
            args.model_dir,
            action_dim=args.vla_action_dim,
            action_horizon=args.vla_action_horizon,
            num_denoise_steps=args.vla_num_denoise_steps,
            robot_adapter=load_optional_json(args.vla_robot_adapter, "robot adapter"),
            norm_config=load_optional_json(args.vla_norm_config, "normalization config"),
            action_postprocess_config=load_optional_json(
                args.vla_action_postprocess_config, "action postprocess config"
            ),
            dtype=args.data_type,
        )
