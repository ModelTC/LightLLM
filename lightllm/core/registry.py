from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AttentionProvider:
    create: Callable
    phases: tuple[str, ...] = ("prefill", "decode", "verify")


@dataclass(frozen=True)
class StateProvider:
    layout: str
    retention: str
    create_store: Callable
    create_writer: Callable | None = None
    supports: Callable | None = None


@dataclass(frozen=True)
class TransferProvider:
    supports: Callable
    bind: Callable


class ProviderRegistry:
    def __init__(
        self,
        attention=(),
        storage=(),
        *,
        static_params=None,
        parameters=(),
        linear=(),
        moe=(),
        expert=(),
        weight_finalizers=(),
        operations=(),
        transfers=(),
    ):
        self.transfers = tuple(transfers)
        self.operations = dict(operations)
        self.weight_finalizers = tuple(weight_finalizers)
        self.attention = tuple(attention)
        self.storage = tuple(storage)
        self.parameters = dict(parameters)
        self._static_params = static_params
        self.linear, self.moe, self.expert = tuple(linear), tuple(moe), tuple(expert)

    def bind_operation(self, name, spec, resources, device, communication=()):
        bind = self.operations.get(name)
        return None if bind is None else bind(spec, resources, device, communication)

    def transfer_driver(self, request, runtime):
        matches = [provider for provider in self.transfers if provider.supports(request, runtime.caps)]
        if not matches:
            raise ValueError(f"unsupported transfer path: {request}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous transfer providers: {request}")
        return matches[0].bind(request, runtime)

    def static_params(self, spec, device, weight=None):
        for kind in type(spec).__mro__:
            if kind in self.parameters:
                return self.parameters[kind](spec, weight, device=device)
        if self._static_params is not None:
            return self._static_params(spec, device)
        raise ValueError(f"static parameter provider is not registered: {type(spec).__name__}")

    @staticmethod
    def validate_attention_requests(requests, backends):
        for request in requests:
            if request.requested == "auto":
                continue
            if not any(
                backend.phase == request.phase
                and request.requested in (backend.name, getattr(backend, "implementation", None))
                for backend in backends
            ):
                raise ValueError(f"{request.phase} attention backend is not registered: {request.requested}")

    def attention_backends(self, phases):
        return tuple(
            provider.create(phase) for provider in self.attention for phase in phases if phase in provider.phases
        )

    def state_provider(self, format, retention):
        matches = [
            provider
            for provider in self.storage
            if provider.layout == format.layout
            and provider.retention == retention
            and (provider.supports is None or provider.supports(format))
        ]
        identity = format.layout, retention, format.encoding
        if not matches:
            raise ValueError(f"state provider is not registered: {identity}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous state providers: {identity}")
        return matches[0]


def default_providers():
    from lightllm.core.compute.attention.dense.fa3 import FA3AttentionBackend
    from lightllm.core.compute.attention.dense.triton import TritonAttentionBackend
    from lightllm.core.compute.attention.dense.backend import DenseStaticParams
    from lightllm.core.compute.attention.dense.group_quantized import GroupQuantizedAttentionBackend
    from lightllm.core.compute.attention.dense.static_fp8 import StaticFP8AttentionBackend
    from lightllm.core.compute.attention.dense.flashinfer import FlashInferAttentionBackend
    from lightllm.core.compute.attention.gdn.backend import GatedDeltaNetBackend
    from lightllm.core.compute.attention.kda.backend import KimiDeltaAttentionBackend
    from lightllm.core.compute.attention.mla.backend import MLAAttentionBackend
    from lightllm.core.compute.attention.mla.flashinfer import FlashInferMLABackend
    from lightllm.core.compute.attention.sparse_mla.backend import SparseMLAAttentionBackend
    from lightllm.core.compute.attention.pooled_nope.backend import PooledNoPEAttentionBackend
    from lightllm.core.formats.providers import state_providers

    from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.core.compute.attention.gdn.backend import GDNStaticParams
    from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
    from lightllm.core.compute.attention.kda.backend import KDAStaticParams
    from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
    from lightllm.core.compute.attention.mla.backend import MLAStaticParams
    from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
    from lightllm.core.compute.attention.sparse_mla.backend import SparseMLAStaticParams
    from lightllm.core.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
    from lightllm.core.compute.attention.pooled_nope.backend import PooledNoPEStaticParams
    from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
    from lightllm.core.compute.linear.backend import LinearBackend
    from lightllm.core.compute.moe.backend import MoeBackend
    from lightllm.core.compute.moe.expert import ExpertComputeBackend
    from functools import partial

    return ProviderRegistry(
        (
            AttentionProvider(FA3AttentionBackend),
            AttentionProvider(TritonAttentionBackend),
            AttentionProvider(GroupQuantizedAttentionBackend),
            AttentionProvider(StaticFP8AttentionBackend),
            AttentionProvider(FlashInferAttentionBackend),
            *(
                AttentionProvider(lambda phase, name=name: GatedDeltaNetBackend(name, phase))
                for name in ("flashqla", "triton")
            ),
            *(
                AttentionProvider(lambda phase, name=name: MLAAttentionBackend(name, phase))
                for name in ("fa3", "triton")
            ),
            AttentionProvider(FlashInferMLABackend),
            *(
                AttentionProvider(
                    lambda phase, name=name: SparseMLAAttentionBackend(name, phase), ("prefill", "decode")
                )
                for name in ("flashmla_sparse", "triton")
            ),
            AttentionProvider(KimiDeltaAttentionBackend),
            *(
                AttentionProvider(lambda phase, name=name: PooledNoPEAttentionBackend(name, phase))
                for name in ("flashmla_sparse", "triton")
            ),
        ),
        state_providers(),
        parameters=(
            (kind, params.from_weights)
            for kind, params in (
                (DenseAttentionSpec, DenseStaticParams),
                (GatedDeltaNetSpec, GDNStaticParams),
                (KimiDeltaAttentionSpec, KDAStaticParams),
                (MLAAttentionSpec, MLAStaticParams),
                (SparseMLAAttentionSpec, SparseMLAStaticParams),
                (PooledNoPEAttentionSpec, PooledNoPEStaticParams),
            )
        ),
        linear=(partial(LinearBackend, name) for name in LinearBackend.encodings),
        moe=(MoeBackend,),
        expert=(ExpertComputeBackend,),
    )
