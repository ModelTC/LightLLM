import datetime
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_reductions(rank, rendezvous, cuda=False):
    import os

    from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args

    args = StartArgs()
    args.enable_tpsp_mix_mode = False
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(args))
    get_env_start_args.cache_clear()
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        layer = Qwen4ExpTransformerLayerInfer.__new__(Qwen4ExpTransformerLayerInfer)
        layer.tp_world_size_ = 4
        layer.tp_rank_ = rank
        state = SimpleNamespace(dist_group=dist.group.WORLD)

        contribution = (256.0, 1.0, -256.0, 0.25)[rank]
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows in (1, 3, 8):
                value = torch.full((rows, 16), contribution, dtype=dtype, device=device)
                value[:, 1::2].neg_()
                expected = torch.full_like(value, 1.25)
                expected[:, 1::2].neg_()
                for reduce in (layer._reduce_linear_output, layer._tpsp_reduce):
                    actual = reduce(value.clone(), state)
                    assert actual.dtype == dtype
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        midpoint_parts = (1.0, 2 ** -8, 2 ** -24, 2 ** -24)
        for rows in (1, 3, 256, 512):
            value = torch.full((rows, 16), midpoint_parts[rank], dtype=torch.bfloat16, device=device)
            expected = torch.full_like(value, 1.0078125)
            for reduce in (layer._reduce_linear_output, layer._tpsp_reduce):
                actual = reduce(value.clone(), state)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        value = torch.full((16, 3), contribution, dtype=torch.bfloat16, device=device).t()
        assert layer._reduce_linear_output(value, state) is value
        torch.testing.assert_close(value, torch.full_like(value, 1.25), atol=0, rtol=0)

        args.enable_tpsp_mix_mode = True
        os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(args))
        get_env_start_args.cache_clear()
        for rows in (4, 12):
            common = torch.arange(rows * 16, device=device, dtype=torch.float32).reshape(rows, 16) / 16
            value = (common + rank / 4).to(torch.bfloat16)
            expected = (common * 4 + 1.5).to(value.dtype).chunk(4)[rank]
            actual = layer._tpsp_reduce(value, state)
            assert actual.shape == (rows // 4, 16) and actual.dtype == value.dtype
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if cuda:
            for _ in range(3):
                layer._tpsp_reduce(value, state)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = layer._tpsp_reduce(value, state)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            graph.reset()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
def test_tp_and_sequence_parallel_accumulation(tmp_path):
    mp.spawn(_check_reductions, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4 or not dist.is_nccl_available(), reason="Four CUDA GPUs are required")
def test_cuda_tp_and_sequence_parallel_accumulation(tmp_path):
    mp.spawn(_check_reductions, args=(str(tmp_path / "nccl-rendezvous"), True), nprocs=4, join=True)


def test_single_rank_preserves_tensor():
    from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer

    layer = Qwen4ExpTransformerLayerInfer.__new__(Qwen4ExpTransformerLayerInfer)
    layer.tp_world_size_ = 1
    value = torch.tensor([[1.25]], dtype=torch.bfloat16)
    assert layer._reduce_linear_output(value, None) is value
    assert layer._tpsp_reduce(value, None) is value
