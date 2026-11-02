from dataclasses import dataclass

from .contracts import AttentionSpec, KVFormat, MlaSpec, MlaFormat


@dataclass(frozen=True)
class KernelOffer:
    backend: object
    phase: str
    metadata_key: tuple
    memory_demands: tuple = ()

    def bind(self, spec: AttentionSpec | MlaSpec):
        return self.backend.for_attention_spec(spec)


@dataclass(frozen=True)
class AttentionChoice:
    format: KVFormat | MlaFormat
    prefill: KernelOffer
    decode: KernelOffer
    rejected: tuple[str, ...]


def select_attention(spec, formats, candidates):

    rejected = []
    for format in formats:
        offers = []
        for phase in ("prefill", "decode"):
            for backend in candidates(format, phase):
                try:
                    offer = probe_attention(backend, spec, format, phase)
                except ValueError as error:
                    rejected.append(str(error))
                else:
                    offers.append(offer)
                    break
            else:
                rejected.append(f"no {phase} implementation for {format.encoding}")
                break
        if len(offers) == 2:
            return AttentionChoice(format, *offers, tuple(rejected))
    detail = "; ".join(rejected)
    raise ValueError(f"{spec.model_role} layer {spec.logical_layer}: no compatible attention plan: {detail}")


def probe_attention(backend, spec: AttentionSpec | MlaSpec, format: KVFormat | MlaFormat, phase: str) -> KernelOffer:

    if phase not in ("prefill", "decode"):
        raise ValueError(f"unknown attention phase {phase}")
    if isinstance(spec, MlaSpec):
        if not isinstance(format, MlaFormat) or spec.latent_dim + spec.rope_dim != format.latent_dim:
            raise ValueError(f"layer {spec.logical_layer}: MLA cache does not match its latent and RoPE dimensions")
        reason = backend.mla_support_error(spec, format, phase)
    else:
        if spec.family != "gqa" or (spec.kv_heads, spec.head_dim) != (format.local_heads, format.head_dim):
            raise ValueError(f"layer {spec.logical_layer}: attention shape does not match its KV representation")
        reason = backend.attention_support_error(spec, format, phase)
    if reason:
        raise ValueError(f"{type(backend).__name__} {phase}, {spec.model_role} layer {spec.logical_layer}: {reason}")
    return KernelOffer(
        backend, phase, backend.attention_metadata_key(spec, format, phase), backend.runtime_memory_demands()
    )
