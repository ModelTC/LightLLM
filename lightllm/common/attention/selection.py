import importlib

from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.utils.backend_validator import validate
from lightllm.utils.log_utils import init_logger
from lightllm.platform import Platform

logger = init_logger(__name__)


class BackendSelectionError(RuntimeError):
    def __init__(self, family, phase, failures):
        self.failures = failures
        details = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(f"No usable {family.value} {phase.value} attention backend: {details}")


def load_backend_class(reference):
    if isinstance(reference, type):
        return reference
    module, name = reference.split(":")
    if not module.startswith("lightllm."):
        module = "lightllm.common.attention." + module
    return getattr(importlib.import_module(module), name)


varlen_backends = {
    "fa3": "backends.dense.fa3.varlen:Fa3VarlenAttentionBackend",
    "xformers": "backends.dense.xformers.varlen:XformersVarlenAttentionBackend",
    "sdpa": "backends.dense.sdpa.varlen:SdpaVarlenAttentionBackend",
    "triton": "backends.dense.triton.varlen:TritonVarlenAttentionBackend",
}


def get_varlen_backend_class(name):
    if name not in varlen_backends:
        raise ValueError(f"unknown varlen attention backend: {name}")
    return load_backend_class(varlen_backends[name])


def select_varlen_attention_backend(requested="auto", *, device=None, devices=None, dtype=None, head_dim=80):
    from lightllm.utils.backend_validator import _validate

    import torch

    if isinstance(dtype, str):
        aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        dtype = getattr(torch, aliases.get(dtype, dtype))
    devices = tuple(devices) if devices is not None else (device,)
    if not devices:
        raise ValueError("varlen selection requires at least one device")
    ordered = tuple(varlen_backends) if requested == "auto" else (requested,)
    failures = {}
    for name in ordered:
        if name not in varlen_backends:
            failures[name] = "unknown varlen attention backend"
            continue
        # 在子进程中检查实际 Q/K/V 路径，避免父进程初始化 CUDA 或误用文本 attention 探针。
        if not all(_validate(name, varlen=True, device=d, dtype=dtype, head_dim=head_dim) for d in devices):
            failures[name] = "varlen environment or numerical probe failed on an assigned device"
            continue
        logger.info("Selected %s varlen attention", name)
        return name
    raise BackendSelectionError(AttentionFamily.DENSE, AttentionPhase.PREFILL, failures)


data_type_to_backend = {
    "None": {
        "triton": "backends.dense.triton.backend:TritonAttBackend",
        "fa3": "backends.dense.fa3.backend:Fa3AttBackend",
        "flashinfer": "backends.dense.flashinfer.backend:FlashInferAttBackend",
    },
    "int4kv": {
        "triton": "backends.dense.triton.int4:Int4kvTritonAttBackend",
    },
    "int8kv": {
        "triton": "backends.dense.triton.int8:Int8kvTritonAttBackend",
    },
    "fp8kv_sph": {
        "fa3": "backends.dense.fa3.fp8:Fp8Fa3AttBackend",
    },
    "fp8kv_spt": {
        "flashinfer": "backends.dense.flashinfer.fp8:Fp8FlashInferAttBackend",
    },
}

mla_data_type_to_backend = {
    "None": {
        "triton": "backends.mla.triton.backend:MlaTritonAttBackend",
        "fa3": "backends.mla.fa3.backend:MlaFa3AttBackend",
        "flashinfer": "backends.mla.flashinfer.backend:MlaFlashInferAttBackend",
    },
}

nsa_data_type_to_backend = {
    "None": {
        "flashmla_sparse": "backends.sparse_mla.flashmla.backend:NsaFlashMlaSparseAttBackend",
    },
    "fp8kv_dsa": {
        "flashmla_sparse": "backends.sparse_mla.flashmla.fp8:NsaFlashMlaFp8SparseAttBackend",
    },
}

neo_data_type_to_backend = {
    "None": {
        "triton": "backends.dense.triton.neo:NeoTritonAttBackend",
        "fa3": "backends.dense.fa3.neo:NeoFa3AttBackend",
    }
}


def select_attention_backend(*, family, phase, model, requested="auto", candidates=None, priority=None, specs=None):
    from lightllm.common.basemodel.attention import describe_attention_layers

    args = model.args
    caps = model.device_runtime.caps
    execution = model.attention_execution
    if candidates is None:
        tables = {
            AttentionFamily.DENSE: data_type_to_backend,
            AttentionFamily.MLA: mla_data_type_to_backend,
            AttentionFamily.SPARSE_MLA: nsa_data_type_to_backend,
            AttentionFamily.GDN: {
                "None": {
                    "flashqla": "backends.gdn.flashqla.backend:FlashQlaLinearAttBackend",
                    "triton": "backends.gdn.triton.backend:TritonLinearAttBackend",
                }
            },
            AttentionFamily.KDA: {"None": {"triton": "backends.kda.triton.backend:KDALinearAttBackend"}},
        }
        if caps.platform is not Platform.CUDA:
            tables = {}
        candidates = tables.get(family, {})
    encoding = "None" if family in (AttentionFamily.GDN, AttentionFamily.KDA) else args.llm_kv_type
    available = candidates.get(encoding, {})
    if priority is None:
        if family is AttentionFamily.GDN:
            priority = ["flashqla", "triton"] if phase is AttentionPhase.PREFILL else ["triton"]
        elif family is AttentionFamily.KDA:
            priority = ["triton"]
        elif family is AttentionFamily.SPARSE_MLA:
            priority = ["flashmla_sparse"]
        else:
            priority = ["fa3", "flashinfer", "triton"]
            if phase is AttentionPhase.DECODE and not args.mtp_step:
                priority = ["flashinfer", "fa3", "triton"]
    ordered = [requested] if requested != "auto" else list(priority)
    specs = describe_attention_layers(model, family, phase) if specs is None else specs
    if not specs:
        raise ValueError(f"model has no {family.value} attention layers")
    failures = {}
    for name in ordered:
        if name not in available:
            failures[name] = f"no implementation for cache encoding {encoding}"
            continue
        try:
            backend = load_backend_class(available[name])
            environment = backend.check_environment(caps)
            if not environment.supported:
                failures[name] = environment.reason
                continue
            for spec in specs:
                result = backend.check_support(spec, caps, execution)
                if not result.supported:
                    failures[name] = result.reason
                    break
            if name in failures:
                continue
            for spec in specs:
                if not validate(available[name], device=str(caps.device), dtype=spec.dtype, spec=spec):
                    failures[name] = "provider numerical probe failed on the assigned device/dtype"
                    break
            if name in failures:
                continue
        except (ImportError, OSError) as exc:
            failures[name] = f"provider unavailable: {exc}"
            continue
        logger.info("Selected %s %s attention on %s: %s", family.value, phase.value, caps.device, name)
        _check_rank_selection(family, phase, None)
        return backend
    error = BackendSelectionError(family, phase, failures)
    _check_rank_selection(family, phase, str(error))
    raise error


def _check_rank_selection(family, phase, failure):
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() > 1:
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        rejected = {f"rank {rank}": reason for rank, reason in enumerate(failures) if reason is not None}
        if rejected:
            raise BackendSelectionError(family, phase, rejected)
