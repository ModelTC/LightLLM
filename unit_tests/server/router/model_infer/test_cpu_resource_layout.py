"""CPU service metadata must match the actual model storage and data path."""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from typing import TYPE_CHECKING
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.att_resources.model_helpers import main_linear_state, prepare_model_states
from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel

from lightllm.models import Deepseek2TpPartModel, Deepseek3_2TpPartModel, LlamaTpPartModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.glm5_next.model import Glm5NextTpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.att_resource_config import cpu_cache_namespace
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args, get_unique_server_name
from lightllm.utils.kv_cache_utils import calcu_cpu_cache_meta

if TYPE_CHECKING:
    from collections.abc import Callable
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.utils.kv_cache_utils import CpuKVCacheMeta
    from pathlib import Path


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("kv_allocator_environment"),
]


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for key, value in {
        "LIGHTLLM_UNIQUE_SERVICE_NAME_ID": tmp_path.name,
        "LIGHTLLM_CURRENT_RANK_IN_NODE": "0",
        "LIGHTLLM_CURRENT_DEVICE_ID": "0",
    }.items():
        monkeypatch.setenv(key, value)
    yield
    get_env_start_args.cache_clear()
    get_added_mtp_kv_layer_num.cache_clear()
    get_unique_server_name.cache_clear()
    calcu_cpu_cache_meta.cache_clear()


CASES = [
    ("dense", "None", LlamaTpPartModel),
    ("int4", "int4kv", LlamaTpPartModel),
    ("int8", "int8kv", LlamaTpPartModel),
    ("fp8_head", "fp8kv_sph", LlamaTpPartModel),
    ("fp8_tensor", "fp8kv_spt", LlamaTpPartModel),
    ("mla", "None", Deepseek2TpPartModel),
    ("packed", "None", Deepseek3_2TpPartModel),
    ("packed_fp8", "fp8kv_dsa", Deepseek3_2TpPartModel),
    ("qwen", "None", Qwen3NextTpPartModel),
    ("qwen_head", "fp8kv_sph", Qwen3NextTpPartModel),
    ("qwen_tensor", "fp8kv_spt", Qwen3NextTpPartModel),
    ("glm", "None", Glm5NextTpPartModel),
]


def make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
    tp: int,
    *,
    mtp_layers: int = 0,
):
    hybrid = kind.startswith("qwen") or kind == "glm"
    config = {
        "model_type": "qwen3_5_text" if kind.startswith("qwen") else "glm5_next" if kind == "glm" else "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4 if hybrid else 2,
        "hidden_size": 128,
        "head_dim": 16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "full_attention_interval": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 3,
        "index_kpool": 4,
        "index_head_dim": 128,
        "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        "linear_attn_config": {"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 3, "kda_layers": [0, 1, 2]},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = StartArgs(
        model_dir=str(tmp_path),
        tp=tp,
        dp=1,
        data_type="bfloat16",
        llm_kv_type=encoding,
        page_size=4,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        cpu_cache_token_page_size=8,
        enable_cpu_cache=True,
        cpu_cache_storage_size=0.001,
        linear_att_cache_size=2,
        mtp_mode="vanilla_with_att" if mtp_layers else None,
        mtp_step=mtp_layers,
    )
    if mtp_layers:
        draft_config = dict(config, num_hidden_layers=1, n_layer=1, full_attention_interval=1)
        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        (draft_dir / "config.json").write_text(json.dumps(draft_config))
        args.mtp_draft_model_dir = [str(draft_dir)] * mtp_layers
        draft_cls = Glm5NextMTPModel if kind == "glm" else model_cls
        monkeypatch.setattr("lightllm.models.get_draft_model_class", lambda *_: draft_cls)
    if encoding in ("fp8kv_sph", "fp8kv_spt"):
        per_head = encoding == "fp8kv_sph"
        calibration_layers = 2 + mtp_layers
        shape = [calibration_layers, 4 if per_head else 2]
        calibration = {
            "num_head": 2,
            "num_layers": calibration_layers,
            "scales_shape": shape,
            "scales": torch.full(shape, 0.125).tolist(),
            "q_calibration": {
                "num_head": 2,
                "scales_shape": [calibration_layers, 2],
                "scales": [[0.25, 0.5]] * calibration_layers,
            },
            "qmin": -448,
            "qmax": 448,
            "quant_type": "per_head" if per_head else "per_tensor",
            "architectures": config["architectures"][0],
        }
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(calibration))
        args.kv_quant_calibration_config_path = str(path)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", str(tp))
    get_env_start_args.cache_clear()
    get_added_mtp_kv_layer_num.cache_clear()
    calcu_cpu_cache_meta.cache_clear()
    monkeypatch.setattr("lightllm.utils.llm_utils.get_llm_model_class", lambda: model_cls)
    meta = calcu_cpu_cache_meta()
    managers = []
    for rank in range(tp):
        monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", str(rank))
        model = model_cls.__new__(model_cls)
        model.args, model.config = args, config
        model.max_total_token_num, model.max_req_num = 16, 1
        model.tp_world_size_, model.tp_rank_ = tp, rank
        prepare_model_states(model)
        drafts = []
        for step in range(mtp_layers):
            draft_cls = Glm5NextMTPModel if kind == "glm" else model_cls
            draft = draft_cls.__new__(draft_cls)
            draft.is_mtp_draft_model = True
            draft.main_model, draft.mtp_previous_draft_models = model, list(drafts)
            draft.args = args
            draft.config = dict(config, num_hidden_layers=1, n_layer=1, full_attention_interval=1)
            draft.max_total_token_num, draft.max_req_num = 16, 1
            draft.tp_world_size_, draft.tp_rank_ = tp, rank
            prepare_model_states(draft, logical_layer_ids=(config["num_hidden_layers"] + step,))
            drafts.append(draft)
        model.att_resource_manager.initialize_resources(
            args, token_capacity=model.max_total_token_num, mem_fraction=args.mem_fraction
        )
        assert not hasattr(model, "mem_manager")
        managers.append(model.att_resource_manager)
    return meta, managers


@pytest.mark.parametrize("kind,encoding,model_cls", CASES)
@pytest.mark.parametrize("tp", [1, 4])
def test_cpu_service_describes_main_and_draft_as_distinct_resources(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
    tp: int,
):
    meta, managers = make_runtime(monkeypatch, tmp_path, kind, encoding, model_cls, tp, mtp_layers=2)
    for resources in managers:
        (main,) = resources.get_full_attention_states("main")
        (draft,) = resources.get_full_attention_states("draft0")
        assert main is not draft
        assert {layer for group in main.config.logical_layer_groups for layer in group}.isdisjoint(
            layer for group in draft.config.logical_layer_groups for layer in group
        )
        assert main.capacity == draft.capacity == resources.full_att_allocator.size
        assert {main.resource_id, draft.resource_id} <= resources.get_cpu_transfer_layouts().keys()
        layout_end = max(layout.byte_end for layout in resources.get_cpu_transfer_layouts().values())
        assert (layout_end + 15) // 16 * 16 == meta.page_bytes
        assert resources.get_cpu_page_bytes() == meta.page_bytes


@pytest.mark.parametrize("kind,encoding,model_cls", CASES)
@pytest.mark.parametrize("tp", [1, 4])
def test_cpu_service_layout_and_roundtrip_all_encodings(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_cache_factory: Callable[[CpuKVCacheMeta], CpuKvCacheClient],
    kind: str,
    encoding: str,
    model_cls: type[TpPartBaseModel],
    tp: int,
):
    meta, managers = make_runtime(monkeypatch, tmp_path, kind, encoding, model_cls, tp)
    # No byte-size formulas duplicated in the test: check the service's actual
    # meta-device descriptors against every real model's layout.
    for resources in managers:
        layout = next(iter(resources.get_cpu_transfer_layouts().values()))
        assert 0 <= layout.byte_start < layout.byte_end <= meta.page_bytes
        assert meta.page_bytes == resources.get_cpu_page_bytes()
    meta.page_num = 3
    client = cpu_cache_factory(meta)
    buffer = client.cpu_kv_cache_tensor
    expected = []
    for resources in managers:
        (full,) = resources.get_full_attention_states("main")
        for field in full_state_test_fields(full, full_state_test_buffers(full)).values():
            if "token" in field.axes:
                field.tensor.fill_(7)
        kwargs = {}
        if resources.bounded_att_big_page_allocator is not None:
            linear = main_linear_state(resources)
            linear.conv_state.fill_(3)
            linear.ssm_state.fill_(5)
            slot = resources.bounded_att_big_page_allocator.alloc(1)[0]
            resources.save_bounded_att_pages([0], page_kind="big", slot_ids=[slot])
            kwargs = dict(big_page_slot_ids=[slot])
        torch.cuda.synchronize()
        expected.append(
            {
                name: f.tensor[:, :8].contiguous().view(torch.uint8).clone()
                for name, f in full_state_test_fields(full, full_state_test_buffers(full)).items()
                if "token" in f.axes
            }
        )
        resources.write_req_to_cpu(torch.arange(8, dtype=torch.long, device="cuda"), buffer, [2], layouts=resources.get_cpu_transfer_layouts(), **kwargs)
    torch.cuda.synchronize()
    for resources, fields in zip(managers, expected):
        (full,) = resources.get_full_attention_states("main")
        for storage_buffer in full_state_test_buffers(full).values():
            storage_buffer[:, 8:16].zero_()
        kwargs = {}
        if resources.bounded_att_big_page_allocator is not None:
            slot = resources.bounded_att_big_page_allocator.alloc(1)[0]
            kwargs = dict(
                big_page_slot_ids=[slot],
                req_idx=1,
            )
        resources.read_req_from_cpu(buffer, [2], torch.arange(8, 16, dtype=torch.long, device="cuda"), layouts=resources.get_cpu_transfer_layouts(), **kwargs)
        if resources.bounded_att_big_page_allocator is not None:
            assert (main_linear_state(resources).conv_state[:, 1] == 3).all()
            assert (main_linear_state(resources).ssm_state[:, 1] == 5).all()
        for name, field in full_state_test_fields(full, full_state_test_buffers(full)).items():
            if "token" in field.axes:
                assert torch.equal(field.tensor[:, 8:16].contiguous().view(torch.uint8), fields[name])
    assert not buffer[:2].any()


def test_cpu_namespace_isolates_model_revision(runtime: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    args = StartArgs(model_dir=str(tmp_path))
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    get_env_start_args.cache_clear()
    (tmp_path / "config.json").write_text("{}")
    initial = cpu_cache_namespace()
    (tmp_path / "model.safetensors").write_bytes(b"first revision")
    updated = cpu_cache_namespace()
    assert updated != initial
    (tmp_path / "model.safetensors").write_bytes(b"second revision")
    assert cpu_cache_namespace() != updated


def test_cpu_namespace_includes_draft_revision(runtime: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    args = StartArgs(model_dir=str(tmp_path), mtp_draft_model_dir=[str(draft_dir)])
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    get_env_start_args.cache_clear()
    before = cpu_cache_namespace()
    (draft_dir / "model.safetensors").write_bytes(b"new draft weights")
    assert cpu_cache_namespace() != before


@pytest.mark.parametrize("hybrid", [False, True])
def test_production_offload_uses_stream_dependency_and_preserves_ready_pages(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_cache_factory: Callable[[CpuKVCacheMeta], CpuKvCacheClient],
    hybrid: bool,
):
    from lightllm.server.multi_level_kv_cache import CacheTier
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common

    kind, encoding, cls = ("qwen", "None", Qwen3NextTpPartModel) if hybrid else ("int4", "int4kv", LlamaTpPartModel)
    meta, (resources,) = make_runtime(monkeypatch, tmp_path, kind, encoding, cls, 1)
    meta.page_num = 2
    client = cpu_cache_factory(meta)
    pages, _ = client.allocate_pages([11, 22], disk_offload_enable=False)
    client.cpu_kv_cache_tensor[pages[0]].fill_(123)
    client.update_pages_status_to_ready([pages[0]], deref=False)
    slots = resources.bounded_att_big_page_allocator.alloc(2) if hybrid else None
    write_batches = []
    write_to_cpu = resources.write_req_to_cpu

    def capture_write(*args: object, **kwargs: object) -> None:
        write_batches.append((kwargs["cpu_page_indices"], kwargs["cpu_page_readies"]))
        write_to_cpu(*args, **kwargs)

    monkeypatch.setattr(resources, "write_req_to_cpu", capture_write)
    if hybrid:
        # 当前 bounded 页仍由主机复制，测试先准备好源快照。
        # TODO：CPU→CPU kernel 落地后，再覆盖尚未完成 D2H 的快照依赖。
        pool = main_linear_state(resources).big_pages_pool
        for slot in slots:
            pool.conv_state[slot].fill_(3)
            pool.ssm_state[slot].fill_(5)
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        # 延迟当前生产流的 KV 写入，验证搬运流通过 wait_stream 等待。
        torch.cuda._sleep(10_000_000)
        (full,) = resources.get_full_attention_states("main")
        for field in full_state_test_fields(full, full_state_test_buffers(full)).values():
            if "token" in field.axes:
                field.tensor.fill_(7)
        req = SimpleNamespace(
            req_idx=0,
            cur_kv_len=16,
            cache_tiers=(CacheTier.CPU,),
            tail_small_page_buffer_id=None,
            hybrid_len_to_big_page_id=dict(zip([8, 16], slots)) if hybrid else {},
            shared_kv_node=None,
            cpu_cache_task_status=InferReq._CpuCacheTaskStatus.NOT_STARTED,
            shm_req=SimpleNamespace(
                token_hash_list=SimpleNamespace(get_all=lambda: [11, 22]),
                token_hash_page_len_list=SimpleNamespace(get_all=lambda: [8, 16]),
            ),
        )
        module = common.MultiLevelKvCacheModule.__new__(common.MultiLevelKvCacheModule)
        module.args = get_env_start_args()
        module.cpu_cache_client = client
        module.gloo_group = None
        module.backend = SimpleNamespace(
            is_master_in_dp=True,
            is_hybrid_att_model=hybrid,
            dp_world_size=1,
            model=SimpleNamespace(
                att_resource_manager=resources,
                req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(16, device="cuda").view(1, -1)),
            ),
        )
        monkeypatch.setattr(common.dist, "broadcast_object_list", lambda *args, **kwargs: None)
        monkeypatch.setattr(common.g_infer_context, "is_hybrid_att_model", hybrid)
        monkeypatch.setattr(common.g_infer_context, "radix_cache", None)
        def reject_host_sync(*args: object, **kwargs: object) -> None:
            pytest.fail("CPU offload must use stream dependencies instead of host synchronization")

        copy_stream = torch.cuda.Stream()
        waited_streams = []
        wait_stream = torch.cuda.Stream.wait_stream

        def record_wait(stream: torch.cuda.Stream, other: torch.cuda.Stream) -> None:
            if stream == copy_stream:
                waited_streams.append(other)
            wait_stream(stream, other)

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda.Stream, "synchronize", reject_host_sync)
            patch.setattr(torch.cuda.Stream, "wait_stream", record_wait)
            task = module.write_req_to_cpu(req, copy_stream)
        assert waited_streams == [producer]
    task.sync_event.synchronize()
    assert write_batches == [(pages, [True, False])]
    assert req.cpu_cache_task_status.is_running()
    assert (client.cpu_kv_cache_tensor[pages[0]] == 123).all()
    for storage_buffer in full_state_test_buffers(full).values():
        storage_buffer[:, :8].zero_()
    kwargs = {}
    if hybrid:
        # 写出已完成，释放测试源快照，给读回目标腾出普通大页槽位。
        req.hybrid_len_to_big_page_id.clear()
        resources.bounded_att_big_page_allocator.free(slots)
        destination = resources.bounded_att_big_page_allocator.alloc(1)[0]
        kwargs = dict(big_page_slot_ids=[destination], req_idx=1)
    resources.read_req_from_cpu(
        client.cpu_kv_cache_tensor, [pages[1]], torch.arange(8, dtype=torch.long, device="cuda"), layouts=resources.get_cpu_transfer_layouts(), **kwargs
    )
    for field in full_state_test_fields(full, full_state_test_buffers(full)).values():
        if "token" in field.axes:
            assert (field.tensor[:, :8] == 7).all()
    if hybrid:
        assert (main_linear_state(resources).conv_state[:, 1] == 3).all()
        assert (main_linear_state(resources).ssm_state[:, 1] == 5).all()


def test_disk_preserves_resource_pages_and_isolates_new_model_revision(
    runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_cache_factory: Callable[[CpuKVCacheMeta], CpuKvCacheClient],
):
    import time

    pytest.importorskip("light_mem")
    from lightllm.server.multi_level_kv_cache.disk_cache_worker import DiskCacheWorker

    meta, (resources,) = make_runtime(monkeypatch, tmp_path, "packed_fp8", "fp8kv_dsa", Deepseek3_2TpPartModel, 1)
    meta.page_num = 2
    client = cpu_cache_factory(meta)
    pages, _ = client.allocate_pages([11, 22], disk_offload_enable=True)
    (full,) = resources.get_full_attention_states("main")
    for storage_buffer in full_state_test_buffers(full).values():
        storage_buffer.random_(0, 256)
    resources.write_req_to_cpu(torch.arange(16, dtype=torch.long, device="cuda"), client.cpu_kv_cache_tensor, pages, layouts=resources.get_cpu_transfer_layouts())
    torch.cuda.synchronize()
    expected = client.cpu_kv_cache_tensor.clone()
    worker = DiskCacheWorker(0.125, client, str(tmp_path / "disk"))
    task = worker.service.create(hash_128s=[11, 22], kv_page_indexer=torch.tensor(pages, dtype=torch.int32), mode="w")
    deadline = time.monotonic() + 10
    while not task.ready():
        assert time.monotonic() < deadline, "resource page disk write did not finish"
        time.sleep(0.01)
    assert all(worker.service.query([11, 22]))
    client.cpu_kv_cache_tensor.zero_()
    assert worker.load_pages([11, 22], pages)
    assert torch.equal(client.cpu_kv_cache_tensor, expected)
    for storage_buffer in full_state_test_buffers(full).values():
        storage_buffer.zero_()
    resources.read_req_from_cpu(client.cpu_kv_cache_tensor, pages, torch.arange(16, dtype=torch.long, device="cuda"), layouts=resources.get_cpu_transfer_layouts())
    # 模型版本改变后使用独立磁盘文件，即使页大小及 token 哈希相同也不命中旧数据。
    (tmp_path / "model.safetensors").write_bytes(b"new model revision")
    next_meta = dataclasses.replace(meta, cache_namespace=cpu_cache_namespace())
    other_client = cpu_cache_factory(next_meta)
    other = DiskCacheWorker(0.125, other_client, str(tmp_path / "disk"))
    assert not any(other.service.query([11, 22]))
    assert any((tmp_path / "disk").glob(f"cache_file_{meta.cache_namespace}*"))
    assert any((tmp_path / "disk").glob(f"cache_file_{next_meta.cache_namespace}*"))
