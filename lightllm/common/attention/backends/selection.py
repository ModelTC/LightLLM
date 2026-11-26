"""各算法族共用的候选加载、构造与依赖校验流程。"""

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch.distributed as dist

from lightllm.utils.log_utils import init_logger
from ..base_backend import AttentionBackend, PrefillAttBackend, DecodeAttBackend

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

logger = init_logger(name=__name__)


def select_backend(
    att_config: Any,
    option: str,
    implementations: Mapping[str, tuple[str, ...]],
    requested: str | None,
    *,
    config_type: type,
    priority: Sequence[str],
    unvalidated: Sequence[str] = (),
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> AttentionBackend:
    """根据指定实现或候选顺序构造后端。

    requested 为 auto 时，按算法族给出的优先级依次尝试，EP 模式跳过 FlashInfer。
    unvalidated 中的实现构造成功即可采用。候选均未通过时，沿用 Triton 回退规则。
    显式指定实现时，直接构造该实现。
    """
    family, phase = option.split(".")
    if not isinstance(att_config, config_type):
        raise NotImplementedError(f"{type(att_config).__name__} is not a {family} attention config")
    from lightllm.utils.backend_validator import validate

    if requested is None:
        requested = model.args.att_backend.get(option, "auto")
    names = tuple(implementations)
    if requested != "auto":
        if requested not in names:
            raise ValueError(f"{phase} attention {requested!r} is not available for {att_config}; choose from {names}")
        impl, reasons = _construct(
            paths=implementations[requested],
            att_config=att_config,
            phase=phase,
            model=model,
            logical_layer_ids=logical_layer_ids,
        )
        if impl is None:
            raise ValueError(f"{phase} attention {requested!r} does not support {att_config}: {'; '.join(reasons)}")
        return impl

    candidates = [
        name for name in priority if name in names and not (model.args.enable_ep_moe and name == "flashinfer")
    ]
    reasons = []
    fallback = None
    for name in dict.fromkeys([*candidates, "triton"]):
        impl, failures = _construct(
            paths=implementations.get(name, ()),
            att_config=att_config,
            phase=phase,
            agree=True,
            model=model,
            logical_layer_ids=logical_layer_ids,
        )
        reasons.extend(failures)
        if impl is None:
            continue
        if name == "triton":
            fallback = impl
        if name not in candidates:
            continue
        if name in unvalidated or validate(backend_name=name):
            logger.info(f"Auto-selected {name} for {phase} attention (validated)")
            return impl
        reasons.append(f"{name}: validation failed")
    if fallback is None:
        raise ValueError(f"no {phase} attention implementation supports {att_config}: {'; '.join(reasons)}")
    logger.warning(f"No validated {phase} attention implementation supports the layer; falling back to triton")
    return fallback


def _load(path: str) -> type[AttentionBackend]:
    module_name, class_name = path.split(":")
    return getattr(import_module(name=module_name), class_name)


def _construction_failures(reason: Optional[str]) -> list[str]:
    """各 rank 在进入验证广播前先就每个候选达成一致。"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [reason] if reason else []
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(object_list=reports, obj=reason)
    if all(report == reports[0] for report in reports):
        return [reports[0]] if reports[0] else []
    return [f"rank {rank}: {report}" for rank, report in enumerate(reports) if report]


def _construct(
    paths: Sequence[str],
    att_config: Any,
    phase: str,
    *,
    agree: bool = False,
    model: "TpPartBaseModel",
    logical_layer_ids: tuple[int, ...],
) -> tuple[AttentionBackend | None, list[str]]:
    """尝试构造支持当前层配置的候选后端，并收集拒绝原因。"""
    expected = PrefillAttBackend if phase == "prefill" else DecodeAttBackend
    reasons = []
    for path in paths:
        impl = None
        reason = None
        label = path
        try:
            backend = _load(path=path)
            label = backend.get_name()
            if not issubclass(backend, expected):
                raise TypeError(f"{backend.__name__} is not a {phase} backend")
            runtime_config = replace(backend.create_runtime_config(model=model), logical_layer_ids=logical_layer_ids)
            impl = backend(att_config=att_config, role=model.att_state_role, runtime_config=runtime_config)
        except (NotImplementedError, ImportError, OSError) as error:
            reason = f"{label}: {error}"
        failures = _construction_failures(reason=reason) if agree else ([reason] if reason else [])
        if not failures:
            return impl, reasons
        reasons.extend(failures)
    return None, reasons
