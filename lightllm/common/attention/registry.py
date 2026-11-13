"""Lazy provider declarations and default preference order; importing this does not load kernels."""

import importlib

from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.platform import Platform


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


VISION_VARLEN_PRIORITY = ("fa3", "xformers", "sdpa", "triton")
AUDIO_VARLEN_PRIORITY = ("fa3", "triton")


def get_varlen_attention_op_class(name):
    if name not in varlen_attention_ops:
        raise ValueError(f"unknown varlen attention operator: {name}")
    return load_backend_class(varlen_attention_ops[name])


cache_format_to_backend = {
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

mla_cache_format_to_backend = {
    "None": {
        "triton": "backends.mla.triton.backend:MlaTritonAttBackend",
        "fa3": "backends.mla.fa3.backend:MlaFa3AttBackend",
        "flashinfer": "backends.mla.flashinfer.backend:MlaFlashInferAttBackend",
    },
}

nsa_cache_format_to_backend = {
    "None": {
        "flashmla_sparse": "backends.sparse_mla.flashmla.backend:NsaFlashMlaSparseAttBackend",
    },
    "fp8kv_dsa": {
        "flashmla_sparse": "backends.sparse_mla.flashmla.fp8:NsaFlashMlaFp8SparseAttBackend",
    },
}

neo_cache_format_to_backend = {
    "None": {
        "triton": "backends.dense.triton.neo:NeoTritonAttBackend",
        "fa3": "backends.dense.fa3.neo:NeoFa3AttBackend",
    }
}


def backend_candidates(family, platform, ssm_state_mode):
    tables = {
        AttentionFamily.DENSE: cache_format_to_backend,
        AttentionFamily.MLA: mla_cache_format_to_backend,
        AttentionFamily.SPARSE_MLA: nsa_cache_format_to_backend,
        AttentionFamily.GDN: {
            "gdn": {
                "flashqla": "backends.gdn.flashqla.backend:FlashQlaGDNBackend",
                "triton": "backends.gdn.triton.backend:TritonGDNBackend",
            }
        },
        AttentionFamily.KDA: {"kda": {"triton": "backends.kda.triton.backend:KDALinearAttBackend"}},
    }
    if ssm_state_mode == "replay":
        tables[AttentionFamily.GDN] = {
            "gdn": {
                "flashqla": "backends.gdn.replay.backend:ReplayGDNFlashQlaAttBackend",
                "triton": "backends.gdn.replay.backend:ReplayGDNTritonAttBackend",
            }
        }
        tables[AttentionFamily.KDA] = {
            "kda": {
                "triton": "backends.kda.replay.backend:ReplayKDALinearAttBackend",
            }
        }
    if platform is not Platform.CUDA:
        tables = {}
    return tables.get(family, {})


def backend_priority(family, phase, specs):
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
    return priority
