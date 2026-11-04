from typing import Any, Protocol
from lightllm.core.contracts.provider import ProbeResult


class AttentionBackend(Protocol):
    name: str

    def probe(self, attention_spec, format_spec, device_caps, limits) -> ProbeResult:
        ...

    def bind(self, offer, state_view, static_params, workspace) -> Any:
        ...
