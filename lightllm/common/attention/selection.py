import importlib

import torch
import torch.distributed as dist

from lightllm.common.attention.api.backend import UnsupportedAttention
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.platform import Platform
from lightllm.utils import backend_validator
from lightllm.utils.backend_validator import validate
from lightllm.utils.log_utils import init_logger

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


varlen_attention_ops = {
    "fa3": "backends.dense.fa3.varlen:Fa3VarlenAttentionOp",
    "xformers": "backends.dense.xformers.varlen:XformersVarlenAttentionOp",
    "sdpa": "backends.dense.sdpa.varlen:SdpaVarlenAttentionOp",
    "triton": "backends.dense.triton.varlen:TritonVarlenAttentionOp",
}


def get_varlen_attention_op_class(name):
    if name not in varlen_attention_ops:
        raise ValueError(f"unknown varlen attention operator: {name}")
    return load_backend_class(varlen_attention_ops[name])


def select_varlen_attention_op(requested="auto", *, device=None, devices=None, dtype=None, head_dim=80):
    if isinstance(dtype, str):
        aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        dtype = getattr(torch, aliases.get(dtype, dtype))
    devices = tuple(devices) if devices is not None else (device,)
    if not devices:
        raise ValueError("varlen selection requires at least one device")
    ordered = tuple(varlen_attention_ops) if requested == "auto" else (requested,)
    failures = {}
    for name in ordered:
        if name not in varlen_attention_ops:
            failures[name] = "unknown varlen attention operator"
            continue
        # 在子进程中检查实际 Q/K/V 路径，避免父进程初始化 CUDA 或误用文本 attention 探针。
        if not all(
            backend_validator._validate(name, varlen=True, device=d, dtype=dtype, head_dim=head_dim) for d in devices
        ):
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


def select_attention_backend(
    *,
    family,
    phase,
    specs,
    caps,
    execution,
    requested="auto",
    candidates=None,
    priority=None,
    encoding="None",
    ssm_state_mode="native",
):
    specs = tuple(specs)
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
        if ssm_state_mode == "replay":
            tables[AttentionFamily.GDN] = {
                "None": {
                    "flashqla": "backends.gdn.replay.backend:ReplayGDNFlashQlaAttBackend",
                    "triton": "backends.gdn.replay.backend:ReplayGDNTritonAttBackend",
                }
            }
            tables[AttentionFamily.KDA] = {
                "None": {
                    "triton": "backends.kda.replay.backend:ReplayKDALinearAttBackend",
                }
            }
        if caps.platform is not Platform.CUDA:
            tables = {}
        candidates = tables.get(family, {})
    encoding = "None" if family in (AttentionFamily.GDN, AttentionFamily.KDA) else encoding
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
            if phase is AttentionPhase.DECODE and not any(spec.use_mtp for spec in specs):
                priority = ["flashinfer", "fa3", "triton"]
    ordered = [requested] if requested != "auto" else list(priority)
    if not specs:
        raise ValueError(f"model has no {family.value} attention layers")
    failures = {}
    selected_backend = None
    selected_name = None
    for name in ordered:
        if name not in available:
            failures[name] = f"no implementation for cache encoding {encoding}"
            continue
        try:
            backend = load_backend_class(available[name])
            backend.check_environment(caps)
            for spec in specs:
                backend.check_support(spec, caps, execution)
            for spec in specs:
                if not validate(available[name], device=str(caps.device), dtype=spec.dtype, spec=spec):
                    failures[name] = "provider numerical probe failed on the assigned device/dtype"
                    break
            if name in failures:
                continue
        except UnsupportedAttention as exc:
            failures[name] = str(exc)
            continue
        except Exception as exc:
            failures[name] = f"provider check failed: {type(exc).__name__}: {exc}"
            continue
        logger.info("Selected %s %s attention on %s: %s", family.value, phase.value, caps.device, name)
        selected_backend = backend
        selected_name = name
        break

    error = None if selected_backend is not None else BackendSelectionError(family, phase, failures)
    _check_rank_selection(family, phase, selected_name, None if error is None else str(error))
    if error is not None:
        raise error
    return selected_backend


def _check_rank_selection(family, phase, selected_name, failure):
    if dist.is_initialized() and dist.get_world_size() > 1:
        selections = [None] * dist.get_world_size()
        dist.all_gather_object(selections, {"backend": selected_name, "failure": failure})
        selected = {item["backend"] for item in selections if item["backend"] is not None}
        rejected = {f"rank {rank}": item["failure"] for rank, item in enumerate(selections) if item["failure"]}
        if len(selected) > 1:
            rejected["backend selection"] = ", ".join(sorted(selected))
        if rejected:
            raise BackendSelectionError(family, phase, rejected)
