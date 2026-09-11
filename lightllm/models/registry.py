import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 定义泛型类型变量，用于保持输入和输出类型的一致性
T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    model_class: Type
    is_multimodal: bool = False
    condition: Optional[Callable[[dict], bool]] = None


def _model_type(model_cfg: dict) -> str:
    return model_cfg.get("model_type") or model_cfg.get("type", "")


def _select_conditional(configs, model_cfg: dict, *, description: str):
    """Select one default/conditional registration for a model config."""

    matches = [
        config
        for config in configs
        if config.condition is None or config.condition(model_cfg)
    ]
    if not matches:
        raise ValueError(f"{description} is not supported")
    if len(matches) > 1:
        matches = [config for config in matches if config.condition is not None]
    if len(matches) != 1:
        raise ValueError(f"ambiguous {description} registration")
    return matches[0]


class _ModelRegistries:
    def __init__(self):
        self._registry: dict[str, list[ModelConfig]] = collections.defaultdict(list)

    def __call__(
        self,
        model_type: Union[str, List[str]],
        is_multimodal: bool = False,
        condition: Optional[Callable[[dict], bool]] = None,
    ) -> Callable[[T], T]:
        """Decorator to register a model (now more concise)."""

        def decorator(
            model_class: T,
        ) -> T:
            model_types = [model_type] if isinstance(model_type, str) else model_type
            for mt in model_types:
                self._registry[mt].append(
                    ModelConfig(
                        model_class=model_class,
                        is_multimodal=is_multimodal,
                        condition=condition,
                    )
                )
            return model_class

        return decorator

    def _select(self, model_cfg: dict) -> ModelConfig:
        model_type = _model_type(model_cfg)
        return _select_conditional(
            self._registry.get(model_type, ()),
            model_cfg,
            description=f"model type {model_type!r}",
        )

    def get_model(self, model_cfg: dict, model_kvargs: dict) -> tuple:
        """Get model"""

        config = self._select(model_cfg)
        return config.model_class(model_kvargs), config.is_multimodal

    def get_model_class(self, model_cfg: dict):
        """Get model"""
        return self._select(model_cfg).model_class


ModelRegistry = _ModelRegistries()


@dataclass(frozen=True)
class GenerationModelConfig:
    """One model-owned implementation of an output-generation capability."""

    model_class: Type
    output_modality: str
    condition: Optional[Callable[[dict], bool]] = None


class _GenerationModelRegistry:
    """Resolve sidecar models without coupling them to target model classes.

    This mirrors ``DraftModelRegistry`` on the LightSpec branch: the common
    runtime knows only a capability name, while each concrete model registers
    the implementation that consumes the target model's Prefix KV.
    """

    def __init__(self):
        self._registry: dict[
            str, list[GenerationModelConfig]
        ] = collections.defaultdict(list)

    def __call__(
        self,
        model_type: Union[str, List[str]],
        output_modality: str,
        condition: Optional[Callable[[dict], bool]] = None,
    ) -> Callable[[T], T]:
        if not output_modality:
            raise ValueError("output modality must not be empty")

        def decorator(model_class: T) -> T:
            model_types = [model_type] if isinstance(model_type, str) else model_type
            for current_model_type in model_types:
                duplicate = any(
                    config.output_modality == output_modality
                    and config.condition is condition
                    for config in self._registry[current_model_type]
                )
                if duplicate:
                    raise ValueError(
                        "duplicate generation model registration: "
                        f"modality={output_modality!r}, "
                        f"model_type={current_model_type!r}"
                    )
                self._registry[current_model_type].append(
                    GenerationModelConfig(
                        model_class=model_class,
                        output_modality=output_modality,
                        condition=condition,
                    )
                )
            return model_class

        return decorator

    def _matches(
        self,
        model_cfg: dict,
        output_modality: Optional[str] = None,
    ) -> List[GenerationModelConfig]:
        matches = [
            config
            for config in self._registry.get(_model_type(model_cfg), ())
            if (output_modality is None or config.output_modality == output_modality)
            and (config.condition is None or config.condition(model_cfg))
        ]
        if len(matches) > 1:
            conditional = [config for config in matches if config.condition is not None]
            if len(conditional) == 1:
                matches = conditional
        return matches

    def get_model_class(self, model_cfg: dict, output_modality: str) -> Type:
        model_type = _model_type(model_cfg)
        matches = self._matches(model_cfg, output_modality)
        if not matches:
            raise ValueError(
                "unsupported generation model: "
                f"modality={output_modality!r}, model_type={model_type!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                "ambiguous generation model registration: "
                f"modality={output_modality!r}, model_type={model_type!r}"
            )
        return matches[0].model_class

    def get_output_modality(self, model_cfg: dict) -> Optional[str]:
        matches = self._matches(model_cfg)
        if not matches:
            return None
        modalities = {config.output_modality for config in matches}
        if len(modalities) != 1 or len(matches) != 1:
            raise ValueError(
                "ambiguous generation capability for "
                f"model_type={_model_type(model_cfg)!r}"
            )
        return matches[0].output_modality


GenerationModelRegistry = _GenerationModelRegistry()


def get_model(model_cfg: dict, model_kvargs: dict):
    try:
        model, is_multimodal = ModelRegistry.get_model(model_cfg, model_kvargs)
        return model, is_multimodal
    except Exception as e:
        logger.exception(str(e))
        raise


def get_model_class(model_cfg: dict):
    try:
        model_class = ModelRegistry.get_model_class(model_cfg)
        return model_class
    except Exception as e:
        logger.exception(str(e))
        raise


def get_generation_model_class(model_cfg: dict, output_modality: str) -> Type:
    return GenerationModelRegistry.get_model_class(model_cfg, output_modality)


def get_output_modality(model_cfg: dict) -> Optional[str]:
    """Return the registered output-generation capability for a target model."""

    return GenerationModelRegistry.get_output_modality(model_cfg)


def is_reward_model() -> Callable[[Dict[str, Any]], bool]:
    """Predicate: whether the model is RewardModel."""
    return lambda model_cfg: "RewardModel" in model_cfg.get("architectures", [""])[0]


def llm_model_type_is(name: Union[str, List[str]]) -> Callable[[Dict[str, Any]], bool]:
    """Predicate: matches model_cfg.get("llm_config").get("model_type") == name."""
    names = [name] if isinstance(name, str) else name
    return lambda model_cfg: (
        model_cfg.get("llm_config", {}).get("model_type", "") in names
        or model_cfg.get("text_config", {}).get("model_type", "") in names
    )
