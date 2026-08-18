"""Registry for mapping target model types and speculative modes to draft models."""

from typing import Callable, Dict, List, Tuple, Type, TypeVar, Union


T = TypeVar("T")


class _DraftModelRegistry:
    def __init__(self):
        self._registry: Dict[Tuple[str, str], Type] = {}

    def __call__(
        self,
        model_type: Union[str, List[str], Tuple[str, ...]],
        spec_modes: Union[str, List[str], Tuple[str, ...]],
    ) -> Callable[[T], T]:
        model_types = (model_type,) if isinstance(model_type, str) else tuple(model_type)
        modes = (spec_modes,) if isinstance(spec_modes, str) else tuple(spec_modes)

        def decorator(model_class: T) -> T:
            for current_model_type in model_types:
                for spec_mode in modes:
                    key = (current_model_type, spec_mode)
                    if key in self._registry:
                        raise ValueError(f"Duplicate draft model registration: {key}")
                    self._registry[key] = model_class
            return model_class

        return decorator

    def get_model_class(self, model_cfg: dict, spec_mode: str) -> Type:
        model_type = model_cfg.get("model_type", "")
        try:
            return self._registry[(model_type, spec_mode)]
        except KeyError:
            raise ValueError(
                f"Unsupported speculative draft model: mode={spec_mode}, model_type={model_type}"
            ) from None


DraftModelRegistry = _DraftModelRegistry()


def get_draft_model_class(model_cfg: dict, spec_mode: str) -> Type:
    return DraftModelRegistry.get_model_class(model_cfg=model_cfg, spec_mode=spec_mode)
