"""Registry mapping draft checkpoint model types and speculative modes to classes."""

from typing import Callable, Dict, List, Tuple, Type, TypeVar, Union

from lightllm.common.mtp_config import FinalHiddenPolicy, MtpPairConfig

from .registry import load_model_class


T = TypeVar("T")


class _DraftModelRegistry:
    def __init__(self):
        self._registry: Dict[Tuple[str, str], MtpPairConfig] = {}

    def __call__(
        self,
        model_type: Union[str, List[str], Tuple[str, ...]],
        spec_modes: Union[str, List[str], Tuple[str, ...]],
        *,
        main_final_hidden_policy: FinalHiddenPolicy,
        draft_final_hidden_policy: FinalHiddenPolicy,
    ) -> Callable[[T], T]:
        def decorator(model_class: T) -> T:
            self.register(
                model_type,
                spec_modes,
                model_class,
                main_final_hidden_policy=main_final_hidden_policy,
                draft_final_hidden_policy=draft_final_hidden_policy,
            )
            return model_class

        return decorator

    def register(
        self,
        model_type: Union[str, List[str], Tuple[str, ...]],
        spec_modes: Union[str, List[str], Tuple[str, ...]],
        model_class: Union[Type, str],
        *,
        main_final_hidden_policy: FinalHiddenPolicy,
        draft_final_hidden_policy: FinalHiddenPolicy,
    ) -> None:
        config = MtpPairConfig(model_class, main_final_hidden_policy, draft_final_hidden_policy)
        model_types = (model_type,) if isinstance(model_type, str) else tuple(model_type)
        modes = (spec_modes,) if isinstance(spec_modes, str) else tuple(spec_modes)
        for current_model_type in model_types:
            for spec_mode in modes:
                key = (current_model_type, spec_mode)
                if key in self._registry:
                    raise ValueError(f"Duplicate draft model registration: {key}")
                self._registry[key] = config

    def get_pair_config(self, model_cfg: dict, spec_mode: str) -> MtpPairConfig:
        model_type = model_cfg.get("model_type", "")
        try:
            return self._registry[(model_type, spec_mode)]
        except KeyError:
            raise ValueError(
                f"Unsupported speculative draft model: mode={spec_mode}, model_type={model_type}"
            ) from None

    def get_model_class(self, model_cfg: dict, spec_mode: str) -> Type:
        return load_model_class(self.get_pair_config(model_cfg, spec_mode).draft_model_class)


DraftModelRegistry = _DraftModelRegistry()


def get_draft_model_class(model_cfg: dict, spec_mode: str) -> Type:
    return DraftModelRegistry.get_model_class(model_cfg=model_cfg, spec_mode=spec_mode)
