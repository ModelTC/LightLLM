"""Attention backend registry and selection.

Backends are registered with ``register_att_backend`` and models pick one with
``resolve_att_backend_class``. A backend is identified by:

* family: the attention algorithm a model layer needs ("dense", "mla", "nsa", "dsv4_nsa", "neo",
  "gdn", "gdn_replay").
* phase: "prefill" or "decode".
* kv type: the ``--llm_kv_type`` cache format it can read.
* name: the value users pass to ``--llm_prefill_att_backend`` / ``--llm_decode_att_backend``.

Registration stores an import path instead of a class, so selecting one backend never imports
the optional GPU dependencies of the others.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, Optional, Tuple, Type, Union, TYPE_CHECKING

from lightllm.utils.backend_validator import validate
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from .base_att import BaseAttBackend

logger = init_logger(__name__)

PREFILL = "prefill"
DECODE = "decode"
# Registered for every --llm_kv_type, for backends that do not read the KV cache format.
ANY_KV_TYPE = "*"
AUTO = "auto"
# Last resort when no candidate passes validation.
_FALLBACK_BACKEND = "triton"

# Auto-selection order per (family, phase). Families without an entry use registration order.
_DEFAULT_PRIORITY: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("dense", PREFILL): ("fa3", "flashinfer", "triton"),
    ("dense", DECODE): ("flashinfer", "fa3", "triton"),
    ("mla", PREFILL): ("fa3", "flashinfer", "triton"),
    ("mla", DECODE): ("flashinfer", "fa3", "triton"),
    ("nsa", PREFILL): ("flashmla_sparse",),
    ("nsa", DECODE): ("flashmla_sparse",),
    ("dsv4_nsa", PREFILL): ("flashmla_sparse",),
    ("dsv4_nsa", DECODE): ("flashmla_sparse",),
    ("neo", PREFILL): ("fa3", "triton"),
    ("gdn", PREFILL): ("flashqla", "triton"),
    ("gdn", DECODE): ("triton",),
    ("gdn_replay", PREFILL): ("flashqla", "triton"),
    ("gdn_replay", DECODE): ("triton",),
}


def _load_attr(ref: str):
    """Import ``"module:attribute"``, with the module relative to this package."""
    module_name, attr_name = ref.split(":")
    return getattr(import_module(f"{__package__}.{module_name}"), attr_name)


@dataclass(frozen=True)
class _BackendEntry:
    backend: Union[str, Type["BaseAttBackend"]]
    needs_validation: bool
    available_flag: Optional[str]
    unavailable_hint: str

    def load(self) -> Type["BaseAttBackend"]:
        return _load_attr(self.backend) if isinstance(self.backend, str) else self.backend

    def available(self) -> bool:
        return self.available_flag is None or bool(_load_attr(self.available_flag))


# (family, phase) -> kv type -> backend name -> entry
_REGISTRY: Dict[Tuple[str, str], Dict[str, Dict[str, _BackendEntry]]] = {}


def register_att_backend(
    family: str,
    name: str,
    backend: Union[str, Type["BaseAttBackend"]],
    *,
    kv_types: Iterable[str] = ("None",),
    phases: Iterable[str] = (PREFILL, DECODE),
    needs_validation: bool = True,
    available_flag: Optional[str] = None,
    unavailable_hint: str = "",
) -> None:
    """Make a backend selectable.

    Args:
        backend: the backend class, or its lazy import path ``"module:Class"`` relative to this package.
        needs_validation: when False, auto selection accepts the backend without running
            ``backend_validator.validate(name)``.
        available_flag: lazy path ``"module:FLAG"`` of a boolean that tells whether an optional
            dependency is installed. An unavailable backend is skipped by auto selection and
            rejected, with ``unavailable_hint``, when requested explicitly.
    """
    phases = tuple(phases)
    for phase in phases:
        if phase not in (PREFILL, DECODE):
            raise ValueError(f"unknown attention phase: {phase}")

    entry = _BackendEntry(backend, needs_validation, available_flag, unavailable_hint)
    for phase in phases:
        for kv_type in kv_types:
            named_entries = _REGISTRY.setdefault((family, phase), {}).setdefault(kv_type, {})
            existing = named_entries.get(name)
            if existing is not None and existing != entry:
                raise ValueError(
                    f"{family} {phase} attention backend '{name}' for kv type '{kv_type}' is already "
                    f"registered as {existing.backend}, cannot register {backend}"
                )
            named_entries[name] = entry


def resolve_att_backend_class(
    family: str,
    phase: str,
    slot_index: int = 0,
    allowed: Optional[Iterable[str]] = None,
) -> Type["BaseAttBackend"]:
    """Pick the backend class for one attention slot of a model.

    Args:
        slot_index: position of this slot in ``--llm_{phase}_att_backend``. A missing position means "auto".
        allowed: backend names the model can run with. It restricts both explicit and auto selection.
    """
    args = get_env_start_args()
    kv_type = args.llm_kv_type
    flag_name = f"--llm_{phase}_att_backend"
    target = f"{family} {phase} attention"

    kv_type_entries = _REGISTRY.get((family, phase), {})
    candidates = {**kv_type_entries.get(ANY_KV_TYPE, {}), **kv_type_entries.get(kv_type, {})}
    if not candidates:
        supported_kv_types = sorted(kv_type_entries)
        raise ValueError(
            f"No {target} backend supports --llm_kv_type {kv_type}; supported kv types: {supported_kv_types}"
        )

    restriction = ""
    if allowed is not None:
        allowed = tuple(allowed)
        candidates = {name: entry for name, entry in candidates.items() if name in allowed}
        restriction = f" (this model only supports {list(allowed)})"

    requested = _get_requested_backend(getattr(args, f"llm_{phase}_att_backend"), slot_index)
    if requested != AUTO:
        entry = candidates.get(requested)
        if entry is None:
            raise ValueError(
                f"{flag_name} value '{requested}' at position {slot_index} is not available for {target} "
                f"with --llm_kv_type {kv_type}{restriction}; valid choices: {[AUTO] + list(candidates)}"
            )
        if not entry.available():
            raise ValueError(
                f"{flag_name} value '{requested}' was requested for {target}, but its dependency is missing.\n"
                f"{entry.unavailable_hint}"
            )
        return entry.load()

    priority = _get_auto_priority(family, phase, candidates, args)
    tried = []
    for name in priority:
        entry = candidates.get(name)
        if entry is None or not entry.available():
            continue
        if not entry.needs_validation:
            logger.info(f"Auto-selected {name} backend for {target}")
            return entry.load()
        tried.append(name)
        if validate(name):
            logger.info(f"Auto-selected {name} backend for {target} (validated)")
            return entry.load()

    fallback = candidates.get(_FALLBACK_BACKEND)
    if fallback is not None and fallback.available():
        logger.warning(f"No backend validation succeeded for {target}, falling back to {_FALLBACK_BACKEND}")
        return fallback.load()
    raise RuntimeError(
        f"No usable {target} backend for --llm_kv_type {kv_type}{restriction}; validation failed for: {tried}"
    )


def _get_requested_backend(backend_args, slot_index: int) -> str:
    return backend_args[slot_index] if len(backend_args) > slot_index else AUTO


def _get_auto_priority(family: str, phase: str, candidates: Dict[str, _BackendEntry], args) -> Tuple[str, ...]:
    priority = _DEFAULT_PRIORITY.get((family, phase)) or tuple(candidates)

    # With MTP, FA3 can make better use of Tensor Core compute and delivers better decode performance.
    if phase == DECODE and args.mtp_step > 0 and "fa3" in priority:
        priority = ("fa3",) + tuple(name for name in priority if name != "fa3")

    if args.enable_ep_moe and "flashinfer" in priority:
        logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")
        priority = tuple(name for name in priority if name != "flashinfer")
    return priority


def _register_builtin_backends() -> None:
    register_att_backend("dense", "triton", "dense.triton:TritonAttBackend")
    register_att_backend("dense", "fa3", "dense.fa3:Fa3AttBackend")
    register_att_backend("dense", "flashinfer", "dense.flashinfer:FlashInferAttBackend")
    register_att_backend("dense", "triton", "dense.triton_int4kv:Int4kvTritonAttBackend", kv_types=("int4kv",))
    register_att_backend("dense", "triton", "dense.triton_int8kv:Int8kvTritonAttBackend", kv_types=("int8kv",))
    register_att_backend("dense", "fa3", "dense.fa3_fp8:Fp8Fa3AttBackend", kv_types=("fp8kv_sph",))
    register_att_backend("dense", "flashinfer", "dense.flashinfer_fp8:Fp8FlashInferAttBackend", kv_types=("fp8kv_spt",))

    register_att_backend("mla", "triton", "mla.triton:MlaTritonAttBackend")
    register_att_backend("mla", "fa3", "mla.fa3:MlaFa3AttBackend")
    register_att_backend("mla", "flashinfer", "mla.flashinfer:MlaFlashInferAttBackend")

    register_att_backend("nsa", "flashmla_sparse", "nsa.flashmla_sparse:NsaFlashMlaSparseAttBackend")
    register_att_backend(
        "nsa",
        "flashmla_sparse",
        "nsa.flashmla_sparse_fp8:NsaFlashMlaFp8SparseAttBackend",
        kv_types=("fp8kv_dsa",),
    )
    register_att_backend(
        "dsv4_nsa",
        "flashmla_sparse",
        "nsa.dsv4_flashmla_sparse_fp8:DeepseekV4FlashMlaFp8SparseAttBackend",
        kv_types=("fp8kv_dsa",),
    )

    register_att_backend("neo", "triton", "neo.triton:NeoTritonAttBackend", phases=(PREFILL,))
    register_att_backend(
        "neo",
        "fa3",
        "neo.fa3:NeoFa3AttBackend",
        phases=(PREFILL,),
        # Only Neo FA3 needs the optional image-token attention interface.
        available_flag="neo.fa3:HAS_FLASH_ATTN_INTERFACE",
        unavailable_hint="Neo FA3 needs the flash_attn_interface build with image_token_end support; "
        "use --llm_prefill_att_backend triton to select Triton instead.",
    )

    # Linear attention does not read the full-attention KV cache format.
    register_att_backend(
        "gdn", "flashqla", "linear.gdn_flashqla:FlashQlaLinearAttBackend", kv_types=(ANY_KV_TYPE,), phases=(PREFILL,)
    )
    register_att_backend(
        "gdn", "triton", "linear.gdn_triton:TritonLinearAttBackend", kv_types=(ANY_KV_TYPE,), needs_validation=False
    )
    register_att_backend(
        "gdn_replay",
        "flashqla",
        "linear.replay_gdn:ReplayGDNFlashQlaAttBackend",
        kv_types=(ANY_KV_TYPE,),
        phases=(PREFILL,),
    )
    register_att_backend(
        "gdn_replay",
        "triton",
        "linear.replay_gdn:ReplayGDNTritonAttBackend",
        kv_types=(ANY_KV_TYPE,),
        needs_validation=False,
    )


_register_builtin_backends()
