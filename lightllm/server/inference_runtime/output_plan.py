from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class OutputKind(str, Enum):
    """A terminal output produced by an inference request."""

    TEXT = "text"
    ACTION = "action"


@dataclass(frozen=True)
class OutputPlan:
    """The output branches requested by one inference request.

    Omitting ``outputs`` preserves LightLLM's legacy text-only behavior.  The
    ``resolve`` helper additionally understands the legacy VLA convention in
    which the presence of an action payload implied action-only output.
    """

    outputs: frozenset[OutputKind] = field(default_factory=lambda: frozenset((OutputKind.TEXT,)))

    def __post_init__(self) -> None:
        values = self.outputs
        if isinstance(values, (str, OutputKind)):
            values = (values,)

        normalized = []
        for value in values:
            try:
                normalized.append(value if isinstance(value, OutputKind) else OutputKind(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported inference output: {value!r}") from exc

        normalized_outputs = frozenset(normalized)
        if not normalized_outputs:
            raise ValueError("an inference request must select at least one output")
        object.__setattr__(self, "outputs", normalized_outputs)

    @classmethod
    def from_outputs(
        cls,
        outputs: OutputPlan | OutputKind | str | Iterable[OutputKind | str] | None = None,
    ) -> OutputPlan:
        if isinstance(outputs, cls):
            return outputs
        if outputs is None:
            return cls.text_only()
        if isinstance(outputs, (str, OutputKind)):
            outputs = (outputs,)
        return cls(frozenset(outputs))

    @classmethod
    def resolve(
        cls,
        outputs: OutputPlan | OutputKind | str | Iterable[OutputKind | str] | None = None,
        *,
        legacy_action_requested: bool = False,
    ) -> OutputPlan:
        """Resolve explicit outputs while preserving both legacy defaults.

        New callers should pass ``outputs``.  During migration, an omitted
        plan means text-only unless the old request shape contains an action
        payload, in which case it means action-only.
        """

        if outputs is None and legacy_action_requested:
            return cls.action_only()
        return cls.from_outputs(outputs)

    @classmethod
    def text_only(cls) -> OutputPlan:
        return cls(frozenset((OutputKind.TEXT,)))

    @classmethod
    def action_only(cls) -> OutputPlan:
        return cls(frozenset((OutputKind.ACTION,)))

    @classmethod
    def text_and_action(cls) -> OutputPlan:
        return cls(frozenset((OutputKind.TEXT, OutputKind.ACTION)))

    @property
    def wants_text(self) -> bool:
        return OutputKind.TEXT in self.outputs

    @property
    def wants_action(self) -> bool:
        return OutputKind.ACTION in self.outputs

    def as_strings(self) -> tuple[str, ...]:
        return tuple(output.value for output in (OutputKind.TEXT, OutputKind.ACTION) if output in self.outputs)

    def __contains__(self, output: OutputKind | str) -> bool:
        try:
            normalized = output if isinstance(output, OutputKind) else OutputKind(output)
        except (TypeError, ValueError):
            return False
        return normalized in self.outputs
