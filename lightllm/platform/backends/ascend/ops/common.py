from collections.abc import Callable
from typing import Any


def unsupported_op(name: str) -> Callable[..., Any]:
    def op(*args, **kwargs):
        raise NotImplementedError(f"Ascend op {name!r} is not implemented.")

    op.__name__ = name.rsplit(".", 1)[-1]
    return op
