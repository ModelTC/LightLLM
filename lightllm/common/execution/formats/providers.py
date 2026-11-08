from lightllm.common.execution.registry import StateProvider
from lightllm.common.execution.contracts.state import StateManager


DENSE_ENCODINGS = (
    ("plain", "None", "dense", "DenseKVWriter"),
    ("int4", "int4kv", "group_quantized", "GroupQuantizedKVWriter"),
    ("int8", "int8kv", "group_quantized", "GroupQuantizedKVWriter"),
    ("fp8_per_head", "fp8kv_sph", "static_fp8", "StaticFP8KVWriter"),
    ("fp8_per_tensor", "fp8kv_spt", "static_fp8", "StaticFP8KVWriter"),
)


def dense_kv_type(format):
    for encoding, kv_type, _, _ in DENSE_ENCODINGS:
        if format.encoding == encoding:
            return kv_type
    raise ValueError(f"unsupported legacy Dense storage encoding: {format.encoding}")


def _require_manager(manager) -> StateManager:
    if manager is None:
        raise ValueError("execution state storage requires a shared StateManager")
    return manager


def recurrent_store(declarations, format, manager, device, specs):
    from lightllm.common.execution.storage.attention import ConvSSMStateStore
    from lightllm.common.execution.compute.attention.kda.spec import KimiDeltaAttentionSpec

    manager = _require_manager(manager)
    kinds = {isinstance(specs[state.producer], KimiDeltaAttentionSpec) for state in declarations}
    if len(kinds) != 1:
        raise ValueError("a recurrent allocation must use one declared state family")
    return ConvSSMStateStore(
        declarations, format, manager, manager.request_capacity, device.device, kind="kda" if kinds.pop() else "gdn"
    )


def window_store(declarations, format, manager, device, specs):
    from lightllm.common.execution.storage.attention import IndexerWindowStateStore

    manager = _require_manager(manager)
    return IndexerWindowStateStore(declarations, format, manager, manager.request_capacity, device.device)


def state_providers():
    from lightllm.common.execution.storage.attention import DenseStateStore, MLAStateStore, PackedMLAStateStore
    from .kv.mla import MLAKVWriter
    from .kv.packed_mla import PackedMLAKVWriter
    from .kv.pooled_nope import PooledNoPEKVWriter

    from importlib import import_module

    providers = []
    for encoding, kv_type, module, name in DENSE_ENCODINGS:
        writer = getattr(import_module(f"lightllm.common.execution.formats.kv.{module}"), name)
        providers.append(
            StateProvider(
                "token_head_dim",
                "history",
                lambda declarations, format, manager, device, specs, kv_type=kv_type: DenseStateStore(
                    declarations, format, manager, kv_type=kv_type
                ),
                writer,
                lambda format, encoding=encoding: format.encoding == encoding,
            )
        )
    for layout, store, writer, encodings in (
        ("token_latent", MLAStateStore, MLAKVWriter, ("plain",)),
        ("token_packed_mla", PackedMLAStateStore, PackedMLAKVWriter, ("plain", "fp8_group128", "fp8_group64_e8m0")),
        ("token_pooled_nope", PackedMLAStateStore, PooledNoPEKVWriter, ("plain", "fp8_group64_e8m0")),
    ):
        providers.append(
            StateProvider(
                layout,
                "history",
                lambda declarations, format, manager, device, specs, store=store: store(declarations, format, manager),
                writer,
                lambda format, encodings=encodings: format.encoding in encodings,
            )
        )
    providers.append(
        StateProvider("request_conv_ssm", "recurrent", recurrent_store, supports=lambda fmt: fmt.encoding == "conv_ssm")
    )
    providers.append(
        StateProvider("request_indexer_window", "window", window_store, supports=lambda fmt: fmt.encoding == "plain")
    )
    return tuple(providers)
