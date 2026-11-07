from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSMRunConfig:
    BV: int
    num_warps: int
    num_stages: int = 3

    def __post_init__(self):
        for value in (self.BV, self.num_warps, self.num_stages):
            if type(value) is not int:
                raise TypeError("SSM kernel configuration must be an integer")
