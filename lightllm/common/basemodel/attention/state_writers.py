from functools import singledispatch

from lightllm.common.attention_state.full_att.compressed_att.compressed_att_state import CompressedAttentionState
from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAFullAttentionState
from lightllm.common.attention_state.full_att.gqa.group_quantized_state import GroupQuantizedFullAttentionState
from lightllm.common.attention_state.full_att.gqa.static_fp8_state import StaticFP8FullAttentionState
from lightllm.common.attention_state.full_att.mla.mla_state import MLAFullAttentionState
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_state import FP8E8M0PackedMLAState
from lightllm.common.attention_state.full_att.packed_mla.packed_mla_state import PackedMLAFullAttentionState


@singledispatch
def bind_kv_writer(state, logical_layer_id):
    raise NotImplementedError(f"no KV writer for {type(state).__name__}")


@bind_kv_writer.register(CompressedAttentionState)
def _bind_compressor(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import fused_compressor_write_only

    return fused_compressor_write_only


@bind_kv_writer.register(GQAFullAttentionState)
def _bind_gqa(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_plain_kv_writer

    return bind_plain_kv_writer(*self.get_layer_view(logical_layer_id))


@bind_kv_writer.register(GroupQuantizedFullAttentionState)
def _bind_group_quantized(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_group_quantized_writer

    return bind_group_quantized_writer(*self.get_layer_view(logical_layer_id), self.config.bits, self.config.group_size)


@bind_kv_writer.register(StaticFP8FullAttentionState)
def _bind_static_fp8(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_static_fp8_writer

    row = self.get_storage_layer_index(logical_layer_id)
    return bind_static_fp8_writer(
        self.k_storage[row], self.v_storage[row], self.scales[row], self.config.scale_mode == "per_tensor"
    )


@bind_kv_writer.register(MLAFullAttentionState)
def _bind_mla(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_latent_writer

    return bind_latent_writer(self.get_layer_view(logical_layer_id))


@bind_kv_writer.register(FP8E8M0PackedMLAState)
def _bind_e8m0(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_e8m0_writer

    return bind_e8m0_writer(self.get_layer_view(logical_layer_id).kv)


@bind_kv_writer.register(PackedMLAFullAttentionState)
def _bind_packed_mla(self, logical_layer_id: int):
    from lightllm.common.basemodel.attention.writers import bind_latent_writer, bind_packed_fp8_writer

    cache = self.get_layer_view(logical_layer_id).kv
    if self.config.kv_encoding == "fp8_group128":
        return bind_packed_fp8_writer(
            cache, self.config.latent_dim - self.config.rope_dim, self.config.rope_dim, self.config.quant_group_size
        )
    return bind_latent_writer(cache)
