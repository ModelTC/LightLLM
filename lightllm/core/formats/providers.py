from lightllm.core.registry import StateProvider


def dense_writer(view):
    from .kv.dense import DenseKVWriter
    from .kv.group_quantized import GroupQuantizedKVWriter
    from .kv.static_fp8 import StaticFP8KVWriter

    return {
        "plain": DenseKVWriter,
        "int4": GroupQuantizedKVWriter,
        "int8": GroupQuantizedKVWriter,
        "fp8_per_head": StaticFP8KVWriter,
        "fp8_per_tensor": StaticFP8KVWriter,
    }[view.format.encoding](view)


def recurrent_store(declarations, format, manager, device, specs):
    from lightllm.core.storage.attention import ConvSSMStateStore
    from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec

    kinds = {isinstance(specs[state.producer], KimiDeltaAttentionSpec) for state in declarations}
    if len(kinds) != 1:
        raise ValueError("a recurrent allocation must use one declared state family")
    return ConvSSMStateStore(
        declarations, format, manager, manager._request_capacity, device.device, kind="kda" if kinds.pop() else "gdn"
    )


def window_store(declarations, format, manager, device, specs):
    from lightllm.core.storage.attention import IndexerWindowStateStore

    return IndexerWindowStateStore(declarations, format, manager, manager._request_capacity, device.device)


def state_providers():
    from lightllm.core.storage.attention import DenseStateStore, MLAStateStore, PackedMLAStateStore
    from .kv.mla import MLAKVWriter
    from .kv.packed_mla import PackedMLAKVWriter
    from .kv.pooled_nope import PooledNoPEKVWriter

    providers = []
    for layout, store, writer in (
        ("token_head_dim", DenseStateStore, dense_writer),
        ("token_latent", MLAStateStore, MLAKVWriter),
        ("token_packed_mla", PackedMLAStateStore, PackedMLAKVWriter),
        ("token_pooled_nope", PackedMLAStateStore, PooledNoPEKVWriter),
    ):
        providers.append(
            StateProvider(
                layout,
                "history",
                lambda declarations, format, manager, device, specs, store=store: store(declarations, format, manager),
                writer,
            )
        )
    providers.append(StateProvider("request_conv_ssm", "recurrent", recurrent_store))
    providers.append(StateProvider("request_indexer_window", "window", window_store))
    return tuple(providers)
