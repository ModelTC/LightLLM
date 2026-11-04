from dataclasses import dataclass
from typing import Any

from lightllm.core.contracts.state import FormatSpec


@dataclass(frozen=True)
class WeightView:
    format: FormatSpec
    fields: dict[str, Any]
