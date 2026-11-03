import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from lightllm.common.weight_cache import cache_authkey, weight_cache_key
from lightllm.server.weight_cache import control_owners

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires a GPU")


@pytest.mark.parametrize(
    "config,explicit_dtype,expected",
    [
        ({"torch_dtype": "bfloat16"}, None, "bfloat16"),
        ({"text_config": {"dtype": "float32"}}, None, "float32"),
        ({}, None, "float16"),
        ({"torch_dtype": "bfloat16"}, "float32", "float32"),
    ],
)
def test_owner_resolves_dtype_before_spawning(tmp_path, monkeypatch, config, explicit_dtype, expected):
    from types import SimpleNamespace
    from lightllm.server import weight_cache

    (tmp_path / "config.json").write_text(json.dumps(config))
    argv = ["weight_cache", "--model_dir", str(tmp_path), "--weight_cache_dir", str(tmp_path / "cache")]
    if explicit_dtype is not None:
        argv.extend(["--data_type", explicit_dtype])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    class ReachedModelWorkers(Exception):
        pass

    def process(*, target, args):
        assert target is weight_cache._run_owner
        assert args[0].data_type == expected
        raise ReachedModelWorkers

    monkeypatch.setattr(weight_cache.mp, "get_context", lambda _: SimpleNamespace(Process=process))
    with pytest.raises(ReachedModelWorkers):
        weight_cache.main()


def _run_cached_model(checkpoint, cache, output):
    """A fresh engine process exercising the existing full model/DSpark/graph test."""
    import pytest
    from unit_tests.models.deepseek_v41 import test_v41_model as cases
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.common.basemodel import basemodel
    from lightllm.common.weight_cache import WeightCacheClient

    with pytest.MonkeyPatch.context() as patch:
        cases.environment.__wrapped__(patch)
        original_init = StartArgs.__init__

        def initialize(self, *args, **kwargs):
            kwargs["weight_cache_dir"] = cache
            original_init(self, *args, **kwargs)

        patch.setattr(StartArgs, "__init__", initialize)
        allocations = []
        original_attach = WeightCacheClient.__init__
        original_skeleton = WeightCacheClient.initialize_model

        def attach(self, *args, **kwargs):
            before = torch.cuda.memory_allocated()
            original_attach(self, *args, **kwargs)
            allocations.append(torch.cuda.memory_allocated() - before)

        @contextmanager
        def skeleton(self):
            before = torch.cuda.memory_allocated()
            with original_skeleton(self):
                yield
            allocations.append(torch.cuda.memory_allocated() - before)

        patch.setattr(WeightCacheClient, "__init__", attach)
        patch.setattr(WeightCacheClient, "initialize_model", skeleton)
        patch.setattr(
            basemodel, "load_hf_weights", lambda *args, **kwargs: pytest.fail("Warm engine reread checkpoint weights")
        )
        cases.test_text_model_constructs_and_chunked_prefill_matches_full(
            Path(checkpoint), patch, 1, "fp8", 128, True, True, [1, 3]
        )
        assert allocations and all(value == 0 for value in allocations), allocations
        Path(output).write_text(json.dumps(dict(ok=True, weight_allocation_deltas=allocations)))


def test_real_owner_survives_two_fresh_model_workers(tmp_path, monkeypatch):
    from unit_tests.models.deepseek_v41 import test_v41_model as cases
    from lightllm.common.basemodel import basemodel
    from safetensors.torch import save_file

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = {}
    original_load = basemodel.load_hf_weights

    def capture(*args, **kwargs):
        captured.update({name: tensor.cpu().contiguous().clone() for name, tensor in kwargs["weight_dict"].items()})
        return original_load(*args, **kwargs)

    with monkeypatch.context() as patch:
        cases.environment.__wrapped__(patch)
        patch.setattr(basemodel, "load_hf_weights", capture)
        cases.test_text_model_constructs_and_chunked_prefill_matches_full(
            checkpoint, patch, 1, "fp8", 128, True, True, [1, 3]
        )
    save_file(captured, checkpoint / "model.safetensors")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in captured}})
    )
    captured.clear()
    gc.collect()
    torch.cuda.empty_cache()
    cache = tmp_path / "cache"
    cache_authkey(cache, create=True)
    command = [
        sys.executable,
        "-m",
        "lightllm.server.weight_cache",
        "--model_dir",
        str(checkpoint),
        "--weight_cache_dir",
        str(cache),
        "--tp",
        "1",
        "--dp",
        "1",
        "--data_type",
        "bfloat16",
        "--mtp_mode",
        "dspark",
        "--mtp_step",
        "3",
        "--mtp_draft_model_dir",
        str(checkpoint),
        "--max_req_total_len",
        "512",
        "--page_size",
        "256",
        "--disable_cudagraph",
    ]
    log = (tmp_path / "owner.log").open("w")
    owner = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    context = torch.multiprocessing.get_context("spawn")
    try:
        deadline = time.monotonic() + 60
        while not list(cache.glob("*.sock")):
            assert owner.poll() is None, (tmp_path / "owner.log").read_text()
            assert time.monotonic() < deadline, (tmp_path / "owner.log").read_text()
            time.sleep(0.2)
        initial = control_owners(cache, "status")
        initial_pid = next(iter(initial.values()))["pid"]
        assert next(iter(initial.values()))["host_tables"]
        for index in range(2):
            output = tmp_path / f"worker_{index}.json"
            process = context.Process(target=_run_cached_model, args=(str(checkpoint), str(cache), str(output)))
            process.start()
            process.join(90)
            if process.is_alive():
                process.kill()
                process.join()
                pytest.fail("Cached worker did not finish")
            assert process.exitcode == 0, (tmp_path / "owner.log").read_text()
            assert json.loads(output.read_text())["ok"]
            current = control_owners(cache, "status")
            assert next(iter(current.values()))["pid"] == initial_pid
            assert next(iter(current.values()))["memory_bytes"] == next(iter(initial.values()))["memory_bytes"]
        deadline = time.monotonic() + 10
        while any(state["clients"] for state in control_owners(cache, "status").values()):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        control_owners(cache, "stop")
        owner.wait(15)
        assert owner.returncode == 0
    finally:
        if owner.poll() is None:
            # Refuse to free weights underneath a still-running test worker.
            control_owners(cache, "stop")
            owner.wait(15)
        log.close()


def test_checkpoint_identity_changes_on_weight_or_layout_options(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from lightllm.server.core.objs.start_args_type import StartArgs

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    save_file({"a": torch.zeros(2)}, tmp_path / "model.safetensors")
    monkeypatch.setenv("LIGHTLLM_GLOBAL_RANK", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    args = StartArgs(model_dir=str(tmp_path))
    before = weight_cache_key(args)
    args.mtp_step = 3
    assert weight_cache_key(args) != before
    args.mtp_step = 0
    save_file({"a": torch.zeros(3)}, tmp_path / "model.safetensors")
    assert weight_cache_key(args) != before


def _small_owner(args):
    from types import SimpleNamespace
    from lightllm.common.basemodel.layer_weights.meta_weights.parameter_weight import ParameterWeight
    from lightllm.server.weight_cache import WeightOwner
    from lightllm.utils.envs_utils import set_env_start_args
    from lightllm.common.weight_cache import cache_socket_path

    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = "0"
    os.environ["LIGHTLLM_GLOBAL_RANK"] = "0"
    os.environ["LIGHTLLM_DP_WORLD_SIZE"] = "1"
    os.environ["LIGHTLLM_CURRENT_RANK_IN_DP"] = "0"
    set_env_start_args(vars(args))
    torch.cuda.set_device(0)
    weight = ParameterWeight("weight", torch.float32, (128, 128))
    weight.weight.fill_(7)
    weight.weight.load_ok = True
    model = SimpleNamespace(named_resident_weights=lambda: {"weight": weight}, trans_layers_weight=[])
    owner = WeightOwner(args, {"target": model}, "test-resident")
    owner.serve(cache_socket_path(args.weight_cache_dir), cache_authkey(args.weight_cache_dir))


def _small_client(args, ready):
    from lightllm.common.weight_cache import WeightCacheClient

    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = "0"
    os.environ["LIGHTLLM_GLOBAL_RANK"] = "0"
    torch.cuda.set_device(0)
    client = WeightCacheClient(args)
    tensor = client.state["target"]["weight"][1]["weight"]
    assert tensor.load_ok and tensor.sum().item() == 7 * 128 * 128
    ready.set()
    time.sleep(60)


def test_owner_rejects_stop_and_worker_exits_on_owner_failure(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from lightllm.server.core.objs.start_args_type import StartArgs

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    save_file({"weight": torch.ones(1)}, tmp_path / "model.safetensors")
    cache = tmp_path / "cache"
    cache_authkey(cache, create=True)
    args = StartArgs(model_dir=str(tmp_path), weight_cache_dir=str(cache))
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_GLOBAL_RANK", "0")
    context = torch.multiprocessing.get_context("spawn")
    owner = context.Process(target=_small_owner, args=(args,))
    ready = context.Event()
    client = context.Process(target=_small_client, args=(args, ready))
    owner.start()
    try:
        deadline = time.monotonic() + 30
        while not list(cache.glob("*.sock")):
            assert owner.is_alive()
            assert time.monotonic() < deadline
            time.sleep(0.1)
        from lightllm.common.weight_cache import WeightCacheClient

        args.mtp_step = 1
        with pytest.raises(RuntimeError, match="fingerprint differs"):
            WeightCacheClient(args)
        args.mtp_step = 0
        assert all(not state["clients"] for state in control_owners(cache, "status").values())
        client.start()
        assert ready.wait(30)
        with pytest.raises(RuntimeError, match="Stop serving workers"):
            control_owners(cache, "stop")
        owner.kill()
        owner.join(15)
        client.join(15)
        assert not client.is_alive()
        assert client.exitcode == 1
    finally:
        for process in (client, owner):
            if process.pid and process.is_alive():
                process.kill()
                process.join(15)


def test_resident_draft_weights_are_not_reserved_twice(monkeypatch):
    from types import SimpleNamespace
    from lightllm.utils import profile_max_tokens

    model = SimpleNamespace(
        max_total_token_num=None,
        is_mtp_draft_model=False,
        args=SimpleNamespace(mtp_mode="dspark"),
        weight_cache_client=object(),
    )
    monkeypatch.setattr(
        profile_max_tokens,
        "get_mtp_adjusted_mem_fraction",
        lambda **kwargs: pytest.fail("All resident target/draft allocations are already included in mem_get_info"),
    )
    with profile_max_tokens.profile_mtp_weight_memory(model):
        pass


def test_owner_builds_actual_ep_shard_without_initializing_collectives(monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
    from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.dist_utils import set_distributed_rank_env
    from lightllm.utils.envs_utils import set_env_start_args, get_env_start_args

    args = StartArgs(tp=16, dp=16, nnodes=2, node_rank=1, enable_ep_moe=True, data_type="bfloat16")
    for key in (
        "GLOBAL_RANK",
        "GLOBAL_WORLD_SIZE",
        "DP_SIZE",
        "DP_WORLD_SIZE",
        "GLOBAL_DP_RANK",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "NODE_WORLD_SIZE",
        "DP_RANK_IN_NODE",
    ):
        monkeypatch.setenv("LIGHTLLM_" + key, "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_START_ARGS", "{}")
    set_env_start_args(vars(args))
    get_env_start_args.cache_clear()

    set_distributed_rank_env(dict(world_size=16, rank_id=8, dp_size=16, args=args))
    assert not torch.distributed.is_initialized()
    weight = FusedMoeWeight(
        gate_proj_name="w1",
        down_proj_name="w2",
        up_proj_name="w3",
        e_score_correction_bias_name="",
        weight_prefix="experts",
        n_routed_experts=32,
        hidden_size=128,
        moe_intermediate_size=128,
        data_type=torch.bfloat16,
        quant_method=DeepGEMMFP8w8a8B128QuantizationMethod(),
        network_config={"num_experts_per_tok": 2},
    )
    assert weight.get_hf_expert_ids() == {16, 17}
    checkpoint = {}
    for expert in (16, 17):
        for matrix in ("w1", "w2", "w3"):
            checkpoint[f"experts.{expert}.{matrix}.weight"] = torch.full((128, 128), expert - 15).to(
                torch.float8_e4m3fn
            )
            checkpoint[f"experts.{expert}.{matrix}.weight_scale_inv"] = torch.ones(1, 1)
    weight.load_hf_weights(checkpoint)
    assert weight.verify_load()
    for index, expert in enumerate((16, 17)):
        assert (weight.w1_list[index].weight.float() == expert - 15).all()
    get_env_start_args.cache_clear()


def _initialize_cache_rank(rank, port, modes):
    from dataclasses import asdict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.dist_utils import init_distributed_env
    from lightllm.utils.envs_utils import set_env_start_args

    args = StartArgs(tp=2, dp=2, nnodes=1, weight_cache_dir="/unused" if modes[rank] else None)
    set_env_start_args(asdict(args))
    try:
        if modes[0] != modes[1]:
            with pytest.raises(ValueError, match="consistently across all model ranks"):
                init_distributed_env(
                    dict(rank_id=rank, world_size=2, dp_size=2, args=args, nccl_host="127.0.0.1", nccl_port=port)
                )
        else:
            init_distributed_env(
                dict(rank_id=rank, world_size=2, dp_size=2, args=args, nccl_host="127.0.0.1", nccl_port=port)
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two GPUs for real startup collective")
@pytest.mark.parametrize("modes", [(True, False), (True, True), (False, False)])
def test_model_ranks_agree_on_resident_cache_mode(modes):
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(_initialize_cache_rank, args=(port, modes), nprocs=2, join=True)
