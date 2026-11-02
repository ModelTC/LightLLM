from dataclasses import dataclass
import json


@dataclass(frozen=True)
class WeightSelection:
    weight_format: str
    activation_format: str
    implementation: str

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in vars(self).values()):
            raise ValueError("weight, activation and implementation names must be nonempty strings")


@dataclass(frozen=True)
class CheckpointFormat:
    encoding: str
    parameters: str

    @classmethod
    def from_config(cls, config):
        return cls("plain" if config is None else config["quant_method"], json.dumps(config, sort_keys=True))


@dataclass(frozen=True)
class WeightPlan:
    checkpoint: CheckpointFormat
    selection: WeightSelection | None
    method_key: str
    use_ue8m0_scales: bool = False

    def bind(self, registry):
        method = registry.get(self.method_key)
        method.hf_quantization_config = json.loads(self.checkpoint.parameters)
        method.use_ue8m0_scales = self.use_ue8m0_scales
        method.weight_plan = self
        return method
