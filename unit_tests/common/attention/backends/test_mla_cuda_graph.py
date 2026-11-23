"""Drive a model's Triton MLA layer group through LightLLM's own CudaGraph capture and replay.

Unlike the step-builder tests that capture with torch.cuda.graph directly, these go through
CudaGraph.capture_decode (warmup on a shallow InferStateInfo copy, graph memory pool) and
CudaGraph.replay (copy_for_cuda_graph into the captured state), with and without decode
microbatch overlap.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.mla import MLADecodeLayerConfig, MLADecodeLayerInputs
from lightllm.common.attention.backends.mla import MLAPrefillLayerConfig
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from unit_tests.common.attention.backends.groups_runner import GroupsRunner


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HEADS, LATENT, ROPE, CAPACITY, BATCH = 16, 512, 64, 192, 2
SCALE = (LATENT + ROPE) ** -0.5


@pytest.fixture
def start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs

    def configure(overlap: bool) -> GroupsRunner:
        """The MLA layers' group of a model running Triton attention."""
        args = {f.name: getattr(StartArgs(), f.name) for f in dataclasses.fields(StartArgs)}
        args.update(enable_decode_microbatch_overlap=overlap, enable_torch_memory_saver=False)
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
        monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
        get_env_start_args.cache_clear()
        args.update(llm_prefill_att_backend=["triton"], llm_decode_att_backend=["triton"], running_max_req_size=BATCH)
        model = SimpleNamespace(
            args=SimpleNamespace(**args),
            max_seq_length=CAPACITY,
            graph_max_len_in_batch=CAPACITY,
            graph_max_batch_size=BATCH,
            tp_world_size_=1,
            is_mtp_draft_model=False,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
        )
        common = dict(num_heads=HEADS, qk_rope_head_dim=ROPE, softmax_scale=SCALE, dtype=torch.bfloat16)
        prefill = MLAPrefillLayerConfig(qk_nope_head_dim=128, v_head_dim=128, **common)
        return GroupsRunner(model, prefill, MLADecodeLayerConfig(kv_lora_rank=LATENT, **common))

    yield configure
    get_env_start_args.cache_clear()


class _Microbatch:
    """Static inputs of one microbatch: its token map, latent cache and two layers' queries."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        slots = BATCH * CAPACITY
        self.token_map = torch.randperm(slots, device="cuda", dtype=torch.int32, generator=generator)
        self.token_map = self.token_map.view(BATCH, CAPACITY)
        self.kv = torch.randn((slots, 1, LATENT + ROPE), device="cuda", dtype=torch.bfloat16, generator=generator)
        # Layer 0 uses the transposed BMM output the model produces; layer 1 a contiguous query.
        self.q_nope = [
            torch.randn((HEADS, BATCH, LATENT), device="cuda", dtype=torch.bfloat16, generator=generator).transpose(
                0, 1
            ),
            torch.randn((BATCH, HEADS, LATENT), device="cuda", dtype=torch.bfloat16, generator=generator),
        ]
        self.q_rope = torch.randn((BATCH, HEADS, ROPE), device="cuda", dtype=torch.bfloat16, generator=generator)

    def state(self, runner, lengths, microbatch_index, *, capture=False, replay=False) -> InferStateInfo:
        state = InferStateInfo()
        state.is_prefill = False
        state.is_cuda_graph_capture, state.is_cuda_graph_replay = capture, replay
        state.input_ids = torch.zeros(BATCH, device="cuda", dtype=torch.int64)
        state.batch_size, state.microbatch_index = BATCH, microbatch_index
        state.b_req_idx = torch.arange(BATCH, device="cuda", dtype=torch.int32)
        state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        state.max_kv_seq_len, state.total_token_num = max(lengths), sum(lengths)
        state.req_manager = SimpleNamespace(req_to_token_indexs=self.token_map)
        runner.step(state)
        self.runner = runner
        return state

    def forward(self, state: InferStateInfo):
        impl = self.runner.decode_impl
        outputs = []
        for q_nope in self.q_nope:
            inputs = MLADecodeLayerInputs(q_nope=q_nope, q_rope=self.q_rope, kv_cache=self.kv)
            out = torch.empty(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
            outputs.append(impl.forward(state.att_states[impl], inputs, out=out))
        return outputs

    def check(self, outputs, lengths) -> None:
        for q_nope, actual in zip(self.q_nope, outputs):
            q = torch.cat((q_nope, self.q_rope), dim=-1).float()
            for i, length in enumerate(lengths):
                rows = self.kv[self.token_map[i, :length].long(), 0].float()
                expected = ((q[i] @ rows.T) * SCALE).softmax(-1) @ rows[:, :LATENT]
                torch.testing.assert_close(actual[i], expected.to(actual.dtype), rtol=0.02, atol=0.012)


def test_decode_graph_replays_new_lengths(start_args):
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    runner = start_args(overlap=False)
    graph = CudaGraph(BATCH, BATCH, BATCH, max_batch_size=BATCH, max_len_in_batch=CAPACITY)
    mb = _Microbatch(seed=0)
    captured = mb.state(runner, (7, 5), 0, capture=True)
    graph.capture_decode(mb.forward, captured)
    for lengths in [(150, 83), (31, 192), (1, 2)]:
        outputs = graph.replay(mb.state(runner, lengths, 0, replay=True))
        mb.check(outputs, lengths)


def test_overlap_graph_keeps_microbatch_schedules_apart(start_args):
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    runner = start_args(overlap=True)
    graph = CudaGraph(BATCH, BATCH, BATCH, max_batch_size=BATCH, max_len_in_batch=CAPACITY)
    mbs = [_Microbatch(seed=1), _Microbatch(seed=2)]

    def forward(state0, state1):
        return mbs[0].forward(state0), mbs[1].forward(state1)

    graph.capture_decode(
        forward, mbs[0].state(runner, (7, 5), 0, capture=True), mbs[1].state(runner, (3, 9), 1, capture=True)
    )
    # Microbatches replay with different lengths, so a shared split schedule would mix them up.
    for lengths0, lengths1 in [((150, 83), (2, 177)), ((40, 40), (190, 1))]:
        outputs0, outputs1 = graph.replay(
            mbs[0].state(runner, lengths0, 0, replay=True), mbs[1].state(runner, lengths1, 1, replay=True)
        )
        mbs[0].check(outputs0, lengths0)
        mbs[1].check(outputs1, lengths1)
