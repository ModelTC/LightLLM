import argparse
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.copy_plan import CopyPlan
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention_state.manifest import build_state_manifest
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.binding import BoundAttention
from lightllm.common.basemodel.attention.contracts import AttentionSpec, ResourceId
from lightllm.common.basemodel.attention.planner import select_attention
from lightllm.common.basemodel.attention.runtime import DeviceRuntime
from lightllm.common.basemodel.attention.torch_cpu import TorchCPUAttentionBackend
from lightllm.utils.envs_utils import get_env_start_args


@dataclass
class Batch:
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor | None
    req_manager: object


def create_storage(groups, dtype):
    manager = AttentionStateManager(request_capacity=2)
    for group in groups:
        state = GQAFullAttentionState(GQAAttentionConfig(group, 2, 16, dtype), device="cpu")
        state.allocate_storage(16)
        state.k_storage.zero_()
        state.v_storage.zero_()
        manager.register_state("main", state)
    manager.finish_state_registration()
    return manager


def bind(manager, backend, layer, dtype):
    state = next(s for s in manager.get_full_attention_states() if s.contains_logical_layer(layer))
    spec = AttentionSpec("main", layer, ResourceId("main", layer), 4, 2, 16, dtype)
    choice = select_attention(spec, (state.get_attention_format(),), lambda format, phase: (backend,))
    provider = choice.prefill.bind(spec)
    cache = state.get_layer_view(layer)
    return BoundAttention(spec, choice.format, cache, provider.bind_cache_writer(cache), AttControl(), 0, 0)


def reference(q, cache, batch):
    expected, row = [], 0
    ready_lengths = batch.b_ready_cache_len
    if ready_lengths is None:
        ready_lengths = batch.b_seq_len - 1
    for request, length, ready in zip(batch.b_req_idx, batch.b_seq_len, ready_lengths):
        for end in range(int(ready) + 1, int(length) + 1):
            slots = batch.req_manager.req_to_token_indexs[request, :end]
            k, v = (plane[slots].float().repeat_interleave(2, 1) for plane in cache)
            scores = torch.einsum("hd,thd->ht", q[row].float(), k) / 4
            expected.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v))
            row += 1
    return torch.stack(expected).to(q.dtype)


def run(dtype):
    runtime = DeviceRuntime(torch.device("cpu"))
    backend = TorchCPUAttentionBackend(object())
    source = create_storage((((0, 2), (1,)),), dtype)
    destination = create_storage((((1,),), ((0, 2),)), dtype)
    assert build_state_manifest(source) == build_state_manifest(destination)
    slots = torch.randperm(16)
    restored_slots = torch.randperm(16)
    mapping = torch.zeros((2, 8), dtype=torch.long)
    mapping[0, :8], mapping[1, :8] = slots[:8], slots[8:]
    batch = Batch(
        torch.tensor([0, 1]), torch.tensor([7, 6]), torch.tensor([3, 4]), SimpleNamespace(req_to_token_indexs=mapping)
    )
    bindings = [bind(source, backend, layer, dtype) for layer in (0, 1)]
    for bound in bindings:
        bound.write_kv(token_indices=slots, kv=torch.randn((16, 4, 16), dtype=dtype))
    state = source.get_full_attention_states()[0]
    assert state.get_layer_view(0)[0].data_ptr() == state.get_layer_view(2)[0].data_ptr()

    page_bytes = source.get_pd_operator().get_page_bytes()
    pages = runtime.allocate((1, page_bytes), torch.uint8)
    CopyPlan.for_pd([source], pages, write=True).run(pd_page_index=0, token_indices=slots)
    source_reusable = runtime.record_completion()
    CopyPlan.for_pd([destination], pages, write=False).run(pd_page_index=0, token_indices=restored_slots)
    destination_ready = runtime.record_completion()
    runtime.wait(destination_ready)
    restored = [bind(destination, backend, layer, dtype) for layer in (0, 1)]
    for before, after in zip(bindings, restored):
        for a, b in zip(before.cache, after.cache):
            assert torch.equal(a[slots].view(torch.uint8), b[restored_slots].view(torch.uint8))

    results = []
    translated = torch.empty(16, dtype=torch.long)
    translated[slots] = restored_slots
    for phase in ("prefill", "decode"):
        if phase == "decode":
            batch.b_seq_len = torch.tensor([8, 7])
            batch.b_ready_cache_len = None
        count = 6 if phase == "prefill" else 2
        q = torch.randn(count, 4, 16, dtype=dtype)
        recovered_batch = Batch(
            batch.b_req_idx,
            batch.b_seq_len,
            batch.b_ready_cache_len,
            SimpleNamespace(req_to_token_indexs=translated[mapping]),
        )
        metadata = backend.create_att_prefill_state(batch)
        recovered_metadata = backend.create_att_prefill_state(recovered_batch)
        metadata.init_state()
        recovered_metadata.init_state()
        for layer, (before, after) in enumerate(zip(bindings, restored)):
            output = getattr(before, phase)(q, (metadata,), runtime.allocate)
            expected = reference(q, before.cache, batch)
            tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
            recovered = getattr(after, phase)(q, (recovered_metadata,), runtime.allocate)
            assert torch.equal(output, recovered)
            results.append(
                dict(
                    phase=phase,
                    layer=layer,
                    query_tokens=count,
                    max_reference_error=(output.float() - expected.float()).abs().max().item(),
                    recovered_bits_equal=True,
                )
            )
    assert source_reusable.query() and destination_ready.query()
    assert not torch.cuda.is_initialized()
    return dict(dtype=str(dtype), page_bytes=page_bytes, checks=results)


def main():
    parser = argparse.ArgumentParser(
        description="Exercise bound attention and native page recovery on CPU, without initializing CUDA."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(dict(pd_kv_page_size=16, cpu_cache_token_page_size=16))
    get_env_start_args.cache_clear()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    result = dict(
        device="cpu",
        machine=platform.machine(),
        torch=torch.__version__,
        cuda_initialized=False,
        results=[run(dtype) for dtype in (torch.float32, torch.bfloat16)],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
