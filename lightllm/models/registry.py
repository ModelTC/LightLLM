import collections
import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 定义泛型类型变量，用于保持输入和输出类型的一致性
T = TypeVar("T")


@dataclass
class ModelConfig:
    model_class: Union[Type, str]
    is_multimodal: bool = False
    condition: Optional[Callable[[dict], bool]] = None
    is_fallback: bool = False


def load_model_class(model_class: Union[Type, str]) -> Type:
    """Resolve a class or a lazy ``module:Class`` reference after selection."""
    if isinstance(model_class, str):
        module_name, class_name = model_class.split(":")
        return getattr(importlib.import_module(module_name), class_name)
    return model_class


class _ModelRegistries:
    def __init__(self):
        self._registry: Dict[str, List[ModelConfig]] = collections.defaultdict(list)

    def __call__(
        self,
        model_type: Union[str, List[str]],
        is_multimodal: bool = False,
        condition: Optional[Callable[[dict], bool]] = None,
        is_fallback: bool = False,
    ) -> Callable[[T], T]:
        """Register an already imported model class as a decorator."""

        def decorator(
            model_class: T,
        ) -> T:
            self.register(
                model_type, model_class, is_multimodal=is_multimodal, condition=condition, is_fallback=is_fallback
            )
            return model_class

        return decorator

    def register(
        self,
        model_type: Union[str, List[str]],
        model_class: Union[Type, str],
        is_multimodal: bool = False,
        condition: Optional[Callable[[dict], bool]] = None,
        is_fallback: bool = False,
    ) -> None:
        """Register a class or lazy path; a fallback can have a condition and still yield to specific matches."""
        model_types = [model_type] if isinstance(model_type, str) else model_type
        for mt in model_types:
            self._registry[mt].append(
                ModelConfig(
                    model_class=model_class, is_multimodal=is_multimodal, condition=condition, is_fallback=is_fallback
                )
            )

    def get_model_config(self, model_cfg: dict) -> ModelConfig:
        """Select metadata; matching conditions take precedence over defaults."""
        model_type = model_cfg.get("model_type", "")
        configs = self._registry.get(model_type, [])
        matches = []
        for cfg in configs:
            if cfg.condition is None or cfg.condition(model_cfg):
                matches.append(cfg)

        if len(matches) == 0:
            raise ValueError(f"Model type {model_type} is not supported.")

        conditional_matches = [m for m in matches if m.condition is not None and not m.is_fallback]
        matches = conditional_matches or matches

        if len(matches) != 1:
            candidates = [m.model_class for m in matches]
            raise ValueError(f"Ambiguous model type {model_type}: {candidates}")
        return matches[0]

    def get_model(self, model_cfg: dict, model_kvargs: dict) -> tuple:
        config = self.get_model_config(model_cfg)
        model_class = load_model_class(config.model_class)
        return model_class(model_kvargs), config.is_multimodal

    def get_model_class(self, model_cfg: dict) -> Type:
        return load_model_class(self.get_model_config(model_cfg).model_class)


ModelRegistry = _ModelRegistries()


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


def is_reward_model() -> Callable[[Dict[str, any]], bool]:
    """Predicate: whether the model is RewardModel."""
    return lambda model_cfg: "RewardModel" in (model_cfg.get("architectures") or [""])[0]


def llm_model_type_is(name: Union[str, List[str]]) -> Callable[[Dict[str, any]], bool]:
    """Predicate: matches model_cfg.get("llm_config").get("model_type") == name."""
    names = [name] if isinstance(name, str) else name
    return lambda model_cfg: (
        (model_cfg.get("llm_config") or {}).get("model_type", "") in names
        or (model_cfg.get("text_config") or {}).get("model_type", "") in names
    )
