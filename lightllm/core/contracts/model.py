from typing import Any, Protocol
from lightllm.core.contracts.state import StateSpec
from lightllm.core.compute.moe.spec import MoeSpec


class ModelAdapter(Protocol):
    attention_phases: tuple[str, ...]

    def moe_specs(self) -> dict[int, MoeSpec]:
        ...

    def attention_specs(self) -> dict[int, Any]:
        ...

    def state_specs(self) -> tuple[StateSpec, ...]:
        ...

    def build_weights(self, quant_config):
        ...

    def build_recipe(self):
        ...

    def load_weights(self, weights, directory, weight_dict=None):
        ...
