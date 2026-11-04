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


class ProviderRegistry:
    def __init__(self, attention=(), storage=(), *, static_params=None):
        self.attention = tuple(attention)
        self.storage = {}
        for provider in storage:
            key = provider.layout, provider.retention
            if key in self.storage:
                raise ValueError(f"duplicate state provider: {key}")
            self.storage[key] = provider
        self.static_params = static_params

    def attention_backends(self, phases):
        return tuple(
            provider.create(phase) for provider in self.attention for phase in phases if phase in provider.phases
        )

    def state_provider(self, format, retention):
        key = format.layout, retention
        if key not in self.storage:
            raise ValueError(f"state provider is not registered: {key}")
        return self.storage[key]


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
        static_params=lambda spec, device: DenseStaticParams(spec, device.allocate_tensor),
    )
