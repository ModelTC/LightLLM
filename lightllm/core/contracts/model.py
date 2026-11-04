from dataclasses import dataclass
from typing import Mapping, Protocol, TypeVar, Generic
from lightllm.core.contracts.state import StateSpec, LayerStates
from lightllm.core.compute.attention.backend import AttentionSpec, AttentionParams
from lightllm.core.contracts.device import DeviceRuntime
from lightllm.core.compute.moe.spec import MoeSpec


PreT = TypeVar("PreT")
PostT = TypeVar("PostT")
LayerT = TypeVar("LayerT")


@dataclass(frozen=True)
class ModelRecipe(Generic[PreT, PostT, LayerT]):
    """Constructed model objects, with layers in forward order.

    Constructors and semantic validation have completed. Weights, runtime
    operations and attention resources must be bound before execution.
    """

    pre: PreT
    post: PostT
    layers: tuple[LayerT, ...]


@dataclass(frozen=True)
class ModelWeights(Generic[PreT, LayerT]):
    pre_post: PreT
    layers: tuple[LayerT, ...]


class ModelAdapter(Protocol):
    attention_phases: tuple[str, ...]

    def moe_specs(self) -> Mapping[int, MoeSpec]:
        ...

    def attention_specs(self) -> Mapping[int, AttentionSpec]:
        ...

    def state_specs(self) -> tuple[StateSpec, ...]:
        ...

    def attention_states(self) -> Mapping[int, LayerStates]:
        ...

    def build_recipe(self) -> ModelRecipe:
        ...

    def build_weights(self, quant_config: object) -> ModelWeights:
        ...

    def load_weights(self, weights: ModelWeights, directory: str, weight_dict: Mapping | None = None) -> None:
        ...

    def attention_params(self, weights: Mapping[int, object], device: DeviceRuntime) -> Mapping[int, AttentionParams]:
        ...
