"""校验启动参数中各算法族公布的后端选项。"""

from collections.abc import Mapping

from lightllm.common.attention.backends.gdn.selector import BACKEND_CHOICES as GDN_CHOICES
from lightllm.common.attention.backends.gqa.selector import BACKEND_CHOICES as GQA_CHOICES
from lightllm.common.attention.backends.kda.selector import BACKEND_CHOICES as KDA_CHOICES
from lightllm.common.attention.backends.mla.selector import BACKEND_CHOICES as MLA_CHOICES
from lightllm.common.attention.backends.sparse_mla.selector import BACKEND_CHOICES as SPARSE_MLA_CHOICES


BACKEND_OVERRIDE_CHOICES = {
    key: ("auto", *implementations)
    for choices in (GQA_CHOICES, MLA_CHOICES, SPARSE_MLA_CHOICES, GDN_CHOICES, KDA_CHOICES)
    for key, implementations in choices.items()
}


def validate_backend_overrides(overrides: Mapping[str, str]) -> None:
    """校验启动参数中指定的算法族和实现。"""
    if not isinstance(overrides, Mapping):
        raise ValueError("expected a mapping of attention family.phase to implementation")
    for key, backend in overrides.items():
        if key not in BACKEND_OVERRIDE_CHOICES:
            raise ValueError(
                f"unknown attention backend key {key!r}; choose from {', '.join(sorted(BACKEND_OVERRIDE_CHOICES))}"
            )
        if backend not in BACKEND_OVERRIDE_CHOICES[key]:
            raise ValueError(
                f"invalid attention backend {backend!r} for {key}; "
                f"choose from {', '.join(BACKEND_OVERRIDE_CHOICES[key])}"
            )
