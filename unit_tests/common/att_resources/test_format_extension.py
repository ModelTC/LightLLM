from lightllm.common.basemodel.attention.state_writers import bind_kv_writer
from lightllm.common.attention_state.storage import allocate_state_storage
from dataclasses import replace
from functools import lru_cache
from math import prod
from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common.attention_state.full_att import FullAttentionState, GQAAttentionConfig
from lightllm.common.attention_state.types import StateTransferLayout, TensorField, TransferRegion
from lightllm.common.attention_state.copy_plan import CopyPlan
from lightllm.common.attention_state.manifest import build_state_manifest, validate_state_manifest
from lightllm.common.basemodel.attention.contracts import KVFormat
from lightllm.common.basemodel.attention.torch_cpu import TorchCPUAttentionBackend, CPUAttentionMetadata
from lightllm.utils.envs_utils import get_env_start_args
from unit_tests.common.basemodel.attention.test_binding import bind_test_model


class AffineInt4State(FullAttentionState):
    storage_fields = ("k", "k_scale", "k_zero", "v", "v_scale", "v_zero")

    def __init__(self):
        self.config = GQAAttentionConfig(((0,),), 2, 16, torch.bfloat16)
        self.page_size, self.state_name = 1, None
        self.device = torch.device("cpu")
        allocate_state_storage(self, 8)

    def get_state_type(self):
        return "test_affine_int4"

    def get_attention_format(self):
        return KVFormat(
            "int4_affine",
            torch.uint8,
            2,
            2,
            16,
            self.page_size,
            0,
            1,
            group_size=8,
            scale_dtype=torch.float32,
            packing="adjacent_low_nibble_first_dynamic_zero",
        )

    def _field_specs(self):
        for plane in ("k", "v"):
            yield plane, torch.uint8, 8
            yield plane + "_scale", torch.float32, 2
            yield plane + "_zero", torch.uint8, 2

    def get_storage_specs(self, token_capacity):
        from lightllm.common.attention_state.storage import TensorStorageSpec

        return tuple(
            TensorStorageSpec(name, (1, token_capacity + self.page_size, 2, width), dtype, fill=0)
            for name, dtype, width in self._field_specs()
        )

    def bind_storage(self, buffers, token_capacity):
        super().bind_storage(buffers, token_capacity)
        self.capacity = token_capacity
        self.fields = {
            name: TensorField(tensor, ("layer", "token", "head", "width")) for name, tensor in buffers.items()
        }

    def get_token_bytes(self):
        return sum(2 * width * dtype.itemsize for _, dtype, width in self._field_specs())

    def contains_logical_layer(self, logical_layer_id):
        return logical_layer_id == 0

    def get_storage_layer_index(self, logical_layer_id):
        assert logical_layer_id == 0
        return 0

    def get_layer_view(self, logical_layer_id):
        row = self.get_storage_layer_index(logical_layer_id)
        return tuple(
            tuple(self.fields[p + suffix].tensor[row] for suffix in ("", "_scale", "_zero")) for p in ("k", "v")
        )

    def get_transfer_fields(self):
        return self.fields

    def _regions(self, tokens):
        regions, offset = [], 0
        for name, dtype, width in self._field_specs():
            shape = (1, tokens, 2, width)
            offset = (offset + dtype.itemsize - 1) // dtype.itemsize * dtype.itemsize
            end = offset + prod(shape) * dtype.itemsize
            regions.append(TransferRegion(name, offset, end, dtype, ("layer", "token", "head", "width"), shape, ()))
            offset = end
        return tuple(regions)

    @lru_cache(None)
    def get_cpu_transfer_regions(self):
        return self._regions(get_env_start_args().cpu_cache_token_page_size)

    @lru_cache(None)
    def get_pd_transfer_regions(self):
        return self._regions(get_env_start_args().pd_kv_page_size)


@bind_kv_writer.register(AffineInt4State)
def bind_affine_writer(self, logical_layer_id):
    cache = self.get_layer_view(logical_layer_id)
    return lambda *, token_indices, kv: affine_write(cache, token_indices, kv)


def affine_write(cache, indices, kv):
    for values, (payload, scales, zeros) in zip(kv.split(2, dim=1), cache):
        groups = values.float().unflatten(-1, (2, 8))
        lo, hi = groups.amin(-1).clamp_max(0), groups.amax(-1).clamp_min(0)
        scale = ((hi - lo) / 15).clamp_min(1e-8)
        zero = (-lo / scale).round().clamp(0, 15)
        codes = (groups / scale[..., None] + zero[..., None]).round().clamp(0, 15).to(torch.uint8).flatten(-2)
        payload[indices] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        scales[indices], zeros[indices] = scale, zero.to(torch.uint8)


def affine_read(cache):
    payload, scales, zeros = cache
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).float().unflatten(-1, (2, 8))
    return ((codes - zeros[..., None]) * scales[..., None]).flatten(-2).to(torch.bfloat16)


class AffineMetadata(CPUAttentionMetadata):
    def prefill_att(self, q, k, v, att_control, alloc_func):
        return super().prefill_att(q, affine_read(k), affine_read(v), att_control, alloc_func)

    decode_att = prefill_att


class AffineBackend(TorchCPUAttentionBackend):
    supported_kv_encodings = ("int4_affine",)
    quant_group_size = 8

    def bind_cache_writer(self, cache):
        return lambda *, token_indices, kv: affine_write(cache, token_indices, kv)

    def create_att_prefill_state(self, infer):
        return AffineMetadata(self, weakref.ref(infer))

    create_att_decode_state = create_att_prefill_state


@pytest.mark.parametrize("path", ["cpu", "pd", "dp", "local"])
def test_new_affine_format_restores_all_fields_and_continues_attention(path, pd_args):
    pd_args.pd_kv_page_size = pd_args.cpu_cache_token_page_size = 8
    torch.manual_seed(15)
    states = [AffineInt4State(), AffineInt4State()]
    models = [bind_test_model([state], AffineBackend(object())) for state in states]
    managers = [model.att_state_manager for model in models]
    src = torch.tensor([7, 2, 0])
    dst = torch.tensor([3, 6, 5])
    if path == "local":
        states[1], models[1], managers[1] = states[0], models[0], managers[0]
    kv = torch.randn(3, 4, 16, dtype=torch.bfloat16)
    models[0].layers_infer[0].write_full_att_kv(token_indices=src, kv=kv)
    validate_state_manifest(*(build_state_manifest(m) for m in managers))
    if path == "pd":
        pages = torch.empty(1, managers[0].get_pd_operator().get_page_bytes(), dtype=torch.uint8)
        for manager, ids, write in zip(managers, (src, dst), (True, False)):
            CopyPlan.for_pd([manager], pages, write).run(pd_page_index=0, token_indices=ids)
    elif path == "cpu":
        operators = [m.get_cpu_cache_operator() for m in managers]
        pages = torch.empty(1, operators[0].get_page_bytes(), dtype=torch.uint8)
        operators[0].write_req(src, pages, [0], operators[0].get_page_layouts())
        operators[1].read_req(pages, [0], dst, operators[1].get_page_layouts())
    elif path == "dp":
        managers[1].get_dp_shared_operator().copy_from_dp(managers[0], src, dst, [], [], 0, 0)
    else:
        managers[0].copy_full_att_tokens(src, dst)
    for name in states[0].fields:
        assert torch.equal(states[0].fields[name].tensor[:, src], states[1].fields[name].tensor[:, dst])
    outputs = []
    q = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    for model, ids in zip(models, (src, dst)):
        layer = model.layers_infer[0]
        layer.alloc_tensor = torch.empty
        infer = SimpleNamespace(
            b_req_idx=torch.tensor([0]),
            b_seq_len=torch.tensor([3]),
            b_ready_cache_len=torch.tensor([2]),
            req_manager=SimpleNamespace(req_to_token_indexs=ids[None, :]),
        )

        class Infer:
            pass

        live = Infer()
        live.__dict__.update(vars(infer))
        metadata = model.prefill_att_backend.create_att_prefill_state(live)
        metadata.init_state()
        live.attention_metadata = (metadata, None)
        outputs.append(layer._token_attention_kernel(q, live, None))
    torch.testing.assert_close(*outputs, atol=0, rtol=0)


def test_manifest_rejects_changed_packing_with_identical_byte_count(pd_args, monkeypatch):
    state = AffineInt4State()
    model = bind_test_model([state], AffineBackend(object()))
    before = build_state_manifest(model.att_state_manager)
    format = state.get_attention_format()
    monkeypatch.setattr(state, "get_attention_format", lambda: replace(format, packing="high_nibble_first"))
    after = build_state_manifest(model.att_state_manager)
    assert before["states"][0]["byte_end"] == after["states"][0]["byte_end"]
    with pytest.raises(ValueError, match="incompatible state semantics"):
        validate_state_manifest(before, after)
