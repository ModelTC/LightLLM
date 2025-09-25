import collections
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

from dataclasses import dataclass
from typing import Type, Dict, Optional, Callable, List, Union, TypeVar
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# 定义泛型类型变量，用于保持输入和输出类型的一致性
T = TypeVar("T")


@dataclass
class ModelConfig:
    model_class: Type
    is_multimodal: bool = False
    condition: Optional[Callable[[dict], bool]] = None


class _ModelRegistries:
    def __init__(self):
        self._registry: Dict[str, List[ModelConfig]] = collections.defaultdict(list)

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
                    ModelConfig(model_class=model_class, is_multimodal=is_multimodal, condition=condition)
                )
            return model_class

        return decorator

    def get_model(self, model_cfg: dict, model_kvargs: dict) -> tuple:
        """Get model"""
        model_type = model_cfg.get("model_type", "")
        configs = self._registry.get(model_type, [])
        matches = []
        for cfg in configs:
            if cfg.condition is None or cfg.condition(model_cfg):
                matches.append(cfg)

        if len(matches) == 0:
            raise ValueError(f"Model type {model_type} is not supported.")

        if len(matches) > 1:
            # Keep conditionally matched models
            matches = [m for m in matches if m.condition is not None]

        assert (
            len(matches) == 1
        ), "Existence of coupled conditon, inability to determine the class of models instantiated"
        model = matches[0].model_class(model_kvargs)
        is_multimodal = matches[0].is_multimodal
        return model, is_multimodal


ModelRegistry = _ModelRegistries()


def get_model(model_cfg: dict, model_kvargs: dict):
    try:
        model, is_multimodal = ModelRegistry.get_model(model_cfg, model_kvargs)
        return model, is_multimodal
    except Exception as e:
        logger.exception(str(e))
        raise


def is_reward_model() -> Callable[[Dict[str, any]], bool]:
    """Predicate: whether the model is RewardModel."""
    return lambda model_cfg: "RewardModel" in model_cfg.get("architectures", [""])[0]


def llm_model_type_is(name: Union[str, List[str]]) -> Callable[[Dict[str, any]], bool]:
    """Predicate: matches model_cfg.get("llm_config").get("model_type") == name."""
    names = [name] if isinstance(name, str) else name
    return lambda model_cfg: (
        model_cfg.get("llm_config", {}).get("model_type", "") in names
        or model_cfg.get("text_config", {}).get("model_type", "") in names
    )
