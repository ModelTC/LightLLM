from dataclasses import dataclass

from .state import FormatSpec


@dataclass(frozen=True)
class TransferRequest:
    source: FormatSpec
    destination: FormatSpec
    source_memory: str
    destination_memory: str
    path: str

    @property
    def requires_conversion(self):
        return self.source != self.destination
