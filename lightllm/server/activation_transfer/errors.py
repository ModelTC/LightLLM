"""Failures that decide whether producer-owned activations may be released."""


class ConsumerExecutionError(RuntimeError):
    def __init__(self, message: str, *, safe_to_release: bool) -> None:
        super().__init__(message)
        self.safe_to_release = safe_to_release


__all__ = ["ConsumerExecutionError"]
