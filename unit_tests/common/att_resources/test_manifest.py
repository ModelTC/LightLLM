from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pickle

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.manifest import (
    build_state_manifest,
    model_content_identity,
    state_manifest_fingerprint,
    validate_state_manifest,
)
from unit_tests.common.att_resources.test_capacity import definition


def manifest(state):
    manager = AttentionStateManager(2)
    manager.register_state("main", state)
    manager.model_identities["main"] = "checkpoint-digest"
    manager.finish_state_registration()
    return build_state_manifest(manager)


def test_equal_page_bytes_do_not_make_different_state_content_compatible(pd_args):
    original = definition("gqa")
    other = type(original)(replace(original.config, logical_layer_groups=((4,), (8,))), device="meta")
    a, b = manifest(original), manifest(other)
    assert a["states"][0]["byte_end"] == b["states"][0]["byte_end"]
    with pytest.raises(ValueError, match="incompatible state semantics"):
        validate_state_manifest(a, b)
    validate_state_manifest(a, deepcopy(a))


def test_recurrent_algorithms_with_equal_shapes_have_different_semantics(pd_args):
    from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState

    config = ConvSSMStateConfig((1, 3), 2, 2, 16, 16, 4, torch.bfloat16, torch.float32)
    a = manifest(GDNState(config, request_capacity=2, device="meta"))
    b = manifest(KDAState(config, request_capacity=2, device="meta"))
    assert a["states"][0]["fields"] == b["states"][0]["fields"]
    with pytest.raises(ValueError, match="incompatible state semantics"):
        validate_state_manifest(a, b)


def test_static_kv_calibration_is_checked_but_query_calibration_is_local(pd_args):
    a, b = definition("static_head"), definition("static_head")
    b.q_scales.add_(1)
    validate_state_manifest(manifest(a), manifest(b))
    b.scales.add_(1)
    with pytest.raises(ValueError, match="calibration"):
        validate_state_manifest(manifest(a), manifest(b))


def test_manifest_rejects_unidentified_peer_and_wrong_model(pd_args):
    a = manifest(definition("gqa"))
    with pytest.raises(ValueError, match="did not provide"):
        validate_state_manifest(a, None)
    b = deepcopy(a)
    b["models"]["main"] = "different-weights"
    with pytest.raises(ValueError, match="model or weight"):
        validate_state_manifest(a, b)


@pytest.mark.parametrize("compact_local", [False, True])
@pytest.mark.parametrize("compact_remote", [False, True])
@pytest.mark.parametrize("change", [None, "version", "models", "calibration", "encoding", "consumers", "fields"])
def test_manifest_fingerprints_preserve_compatibility_checks(pd_args, compact_local, compact_remote, change):
    local = manifest(definition("static_head"))
    remote = deepcopy(local)
    if change == "version":
        remote["version"] += 1
    elif change == "models":
        remote["models"]["main"] = "other checkpoint"
    elif change == "calibration":
        remote["states"][0]["kv_calibration"] = "other calibration"
    elif change == "encoding":
        remote["states"][0]["representation"]["encoding"] = "other encoding"
    elif change == "consumers":
        remote["states"][0]["consumers"].append(20)
    elif change == "fields":
        remote["states"][0]["fields"][0]["layer_index"] += 1
    if compact_local:
        local = state_manifest_fingerprint(local)
    if compact_remote:
        remote = state_manifest_fingerprint(remote)
    if change is None:
        validate_state_manifest(local, remote)
    else:
        with pytest.raises(ValueError):
            validate_state_manifest(local, remote)


@pytest.mark.parametrize("invalid", [None, "", "0" * 63, "g" * 64, 123])
def test_manifest_rejects_malformed_fingerprint(pd_args, invalid):
    local = manifest(definition("gqa"))
    remote = state_manifest_fingerprint(local)
    remote["states_digest"] = invalid
    with pytest.raises(ValueError, match="invalid state manifest digest"):
        validate_state_manifest(local, remote)
    with pytest.raises(ValueError, match="incomplete state manifest"):
        validate_state_manifest(local, {"version": local["version"]})


@pytest.mark.parametrize("backend", ["nccl", "nixl"])
def test_large_hybrid_manifest_fits_request_transfer_parameters(pd_args, backend):
    from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
    from lightllm.server.core.objs.pd_kv_trans_params import PDKVTransParamObj
    from lightllm.server.pd_io_struct import PDDecodeNodeInfo
    from lightllm.server.router.model_infer.mode_backend.pd.nccl_kv_transporter import NcclKVTransporter
    from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter

    manager = AttentionStateManager(2)
    template = definition("gqa")
    config = replace(template.config, logical_layer_groups=tuple((i,) for i in range(0, 256, 4)))
    manager.register_state("main", type(template)(config, device="meta"))
    config = ConvSSMStateConfig(tuple(i for i in range(256) if i % 4), 2, 2, 16, 16, 4, torch.bfloat16, torch.float32)
    manager.register_state("main", GDNState(config, request_capacity=2, device="meta"))
    manager.finish_state_registration()
    local = build_state_manifest(manager)
    assert len(pickle.dumps(local)) > 8192
    cls = NcclKVTransporter if backend == "nccl" else NixlKVTransporter
    transporter = cls.__new__(cls)
    transporter.state_manifest = local
    transporter.num_pages = 16
    transporter.kv_move_buffer = SimpleNamespace(shape=(16, 4096))
    transporter.page_reg_desc = object()
    transporter.nixl_agent = SimpleNamespace(get_serialized_descs=lambda _: b"registered memory" * 32)
    descriptor = transporter.local_page_mem_desc
    assert transporter.local_page_mem_desc is descriptor
    validate_state_manifest(local, pickle.loads(descriptor)["state_manifest"])
    info = PDDecodeNodeInfo(1, 2, "decode_0", b"agent metadata" * 32, 16, descriptor, 3, 0)
    params = PDKVTransParamObj()
    params.set(pickle.dumps(info))
    assert pickle.loads(params.get()) == info
    assert params.data_len < 2048


@pytest.mark.parametrize("window,stride", [(4, 4), (128, 128), (8, 4)])
@pytest.mark.parametrize("target", ["kv", "indexer"])
def test_cpu_manifest_includes_compressor_semantics_without_snapshot_bytes(cpu_args, window, stride, target):
    from lightllm.common.attention_state.bounded_att import WindowCompressorState, WindowCompressorStateConfig

    cpu_args.cpu_cache_token_page_size = 128
    cpu_args.linear_att_hash_page_size = 128
    manager = AttentionStateManager(2)
    manager.register_state("main", definition("gqa"))
    state = WindowCompressorState(
        WindowCompressorStateConfig((1, 3), window, stride, 16, target=target), request_capacity=2, device="meta"
    )
    manager.register_state("main", state)
    manager.finish_state_registration()
    result = build_state_manifest(manager, domain="cpu")
    items = [item for item in result["states"] if item["kind"] == "compressor_" + target]
    assert [item["producer_layer"] for item in items] == [1, 3]
    assert all(bool(item["fields"]) == (window > stride) for item in items)
    if window == stride:
        assert all(item["byte_start"] is item["byte_end"] is item["tokens_per_page"] is None for item in items)
    with pytest.raises(ValueError, match="missing cpu transfer layout"):
        build_state_manifest(manager, {}, domain="cpu")
    with pytest.raises(ValueError, match="missing pd transfer layout"):
        build_state_manifest(manager, manager.get_cpu_cache_operator().get_page_layouts() | {state.state_name: None})


def test_checkpoint_identity_tracks_weight_bytes_without_backend_names(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"first checkpoint")
    model = SimpleNamespace(
        weight_dict=None,
        weight_dir_=str(tmp_path),
        config={"head_dim": 128},
        data_type=torch.bfloat16,
        quant_type="none",
        quant_cfg_path=None,
    )
    before = model_content_identity(model)
    model.prefill_att_backend = "another-compatible-kernel"
    assert model_content_identity(model) == before
    path.write_bytes(b"other checkpoint")
    assert model_content_identity(model) != before


def test_published_pd_identity_rejects_weight_mutation_before_loading():
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    model = SimpleNamespace(state_content_identity="published-checkpoint")
    with pytest.raises(RuntimeError, match="draining transfers"):
        TpPartBaseModel.load_weights(model, {})


@pytest.mark.parametrize("option", ["enable_cpu_cache", "enable_disk_cache", "enable_dp_prompt_cache_fetch"])
def test_bound_external_cache_rejects_weight_mutation_before_loading(option):
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    model = SimpleNamespace(att_state_manager=object(), args=SimpleNamespace(**{option: True}))
    with pytest.raises(RuntimeError, match="rebuilding external caches"):
        TpPartBaseModel.load_weights(model, {})


def test_weight_identity_uses_declared_quantization_semantics(tmp_path):
    from lightllm.common.quantization import Quantcfg

    model = SimpleNamespace(
        weight_dict={"q": torch.ones(1)},
        config={"n_layer": 1},
        data_type=torch.bfloat16,
        quant_cfg=Quantcfg(dict(n_layer=1), "fp8w8a8-b128-triton"),
    )
    original = model_content_identity(model)
    model.quant_cfg = Quantcfg(dict(n_layer=1), "fp8w8a8-b128-vllm")
    assert model_content_identity(model) == original
    model.quant_cfg = Quantcfg(
        dict(n_layer=1), dict(weight_format="fp8_block128", activation_format="fp8_group128", implementation="triton")
    )
    assert model_content_identity(model) == original
    model.quant_cfg = Quantcfg(dict(n_layer=1), "fp8w8a8-pt-triton")
    assert model_content_identity(model) != original
    model.quant_cfg = Quantcfg(dict(n_layer=1), "fp8w8a8-b128-triton")
    model.quant_cfg.quant_cfg[0]["q"] = "none"
    assert model_content_identity(model) != original


def test_disk_namespace_rejects_equal_sized_incompatible_content_without_touching_pages(tmp_path, pd_args):
    from lightllm.common.attention_state.manifest import validate_disk_manifest

    original = manifest(definition("gqa"))
    validate_disk_manifest(tmp_path, original)
    data = tmp_path / "cache_file_0"
    data.write_bytes(b"existing cache pages")
    validate_disk_manifest(tmp_path, deepcopy(original))
    changed = deepcopy(original)
    changed["models"]["main"] = "different-checkpoint"
    with pytest.raises(ValueError, match="content or layout differs"):
        validate_disk_manifest(tmp_path, changed)
    assert data.read_bytes() == b"existing cache pages"


def test_unidentified_disk_pages_are_not_adopted(tmp_path):
    from lightllm.common.attention_state.manifest import validate_disk_manifest

    (tmp_path / "cache_file_0").write_bytes(b"unidentified")
    with pytest.raises(ValueError, match="no state manifest"):
        validate_disk_manifest(tmp_path, dict(version=1))
    assert not (tmp_path / "state_manifest.json").exists()
