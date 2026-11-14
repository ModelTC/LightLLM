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


BACKENDS = (
    "lightllm.common.attention.backends.dense.fa3.backend:Fa3DenseDecodeBackend",
    "lightllm.common.attention.backends.dense.fa3.backend:Fa3DensePrefillBackend",
    "lightllm.common.attention.backends.dense.fa3.fp8:Fa3DenseFp8DecodeBackend",
    "lightllm.common.attention.backends.dense.fa3.fp8:Fa3DenseFp8PrefillBackend",
    "lightllm.common.attention.backends.dense.fa3.neo:Fa3DenseNeoDecodeBackend",
    "lightllm.common.attention.backends.dense.fa3.neo:Fa3DenseNeoPrefillBackend",
    "lightllm.common.attention.backends.dense.flashinfer.backend:FlashInferDenseDecodeBackend",
    "lightllm.common.attention.backends.dense.flashinfer.backend:FlashInferDensePrefillBackend",
    "lightllm.common.attention.backends.dense.flashinfer.fp8:FlashInferDenseFp8DecodeBackend",
    "lightllm.common.attention.backends.dense.flashinfer.fp8:FlashInferDenseFp8PrefillBackend",
    "lightllm.common.attention.backends.dense.triton.backend:TritonDenseDecodeBackend",
    "lightllm.common.attention.backends.dense.triton.backend:TritonDensePrefillBackend",
    "lightllm.common.attention.backends.dense.triton.int4:TritonDenseInt4DecodeBackend",
    "lightllm.common.attention.backends.dense.triton.int4:TritonDenseInt4PrefillBackend",
    "lightllm.common.attention.backends.dense.triton.int8:TritonDenseInt8DecodeBackend",
    "lightllm.common.attention.backends.dense.triton.int8:TritonDenseInt8PrefillBackend",
    "lightllm.common.attention.backends.dense.triton.neo:TritonDenseNeoDecodeBackend",
    "lightllm.common.attention.backends.dense.triton.neo:TritonDenseNeoPrefillBackend",
    "lightllm.common.attention.backends.gdn.flashqla.backend:FlashQlaGDNPrefillBackend",
    "lightllm.common.attention.backends.gdn.replay.backend:FlashQlaGDNReplayPrefillBackend",
    "lightllm.common.attention.backends.gdn.replay.backend:TritonGDNReplayDecodeBackend",
    "lightllm.common.attention.backends.gdn.replay.backend:TritonGDNReplayPrefillBackend",
    "lightllm.common.attention.backends.gdn.triton.backend:TritonGDNDecodeBackend",
    "lightllm.common.attention.backends.gdn.triton.backend:TritonGDNPrefillBackend",
    "lightllm.common.attention.backends.kda.replay.backend:TritonKDAReplayDecodeBackend",
    "lightllm.common.attention.backends.kda.replay.backend:TritonKDAReplayPrefillBackend",
    "lightllm.common.attention.backends.kda.triton.backend:TritonKDADecodeBackend",
    "lightllm.common.attention.backends.kda.triton.backend:TritonKDAPrefillBackend",
    "lightllm.common.attention.backends.mla.fa3.backend:Fa3MLADecodeBackend",
    "lightllm.common.attention.backends.mla.fa3.backend:Fa3MLAPrefillBackend",
    "lightllm.common.attention.backends.mla.flashinfer.backend:FlashInferMLADecodeBackend",
    "lightllm.common.attention.backends.mla.flashinfer.backend:FlashInferMLAPrefillBackend",
    "lightllm.common.attention.backends.mla.triton.backend:TritonMLADecodeBackend",
    "lightllm.common.attention.backends.mla.triton.backend:TritonMLAPrefillBackend",
    "lightllm.common.attention.backends.sparse_mla.flashmla.backend:FlashMLASparseMLADecodeBackend",
    "lightllm.common.attention.backends.sparse_mla.flashmla.backend:FlashMLASparseMLAPrefillBackend",
    "lightllm.common.attention.backends.sparse_mla.flashmla.fp8:FlashMLASparseMLAFp8DecodeBackend",
    "lightllm.common.attention.backends.sparse_mla.flashmla.fp8:FlashMLASparseMLAFp8PrefillBackend",
)


def backend_priority(family, phase, specs):
    if family is AttentionFamily.GDN:
        return ("flashqla", "triton") if phase is AttentionPhase.PREFILL else ("triton",)
    if family is AttentionFamily.KDA:
        return ("triton",)
    if family is AttentionFamily.SPARSE_MLA:
        return ("flashmla_sparse",)
    if phase is AttentionPhase.DECODE and not any(spec.use_mtp for spec in specs):
        return ("flashinfer", "fa3", "triton")
    return ("fa3", "flashinfer", "triton")
