from dataclasses import dataclass


@dataclass(frozen=True)
class ConvSSMStateSpec:
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_history_length: int

    def validate(self):
        if min(self.key_heads, self.value_heads, self.key_dim, self.value_dim, self.conv_history_length) <= 0:
            raise ValueError("Conv/SSM dimensions and history length must be positive")
