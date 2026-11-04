from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput
from lightllm.common.execution.batch_runner import BatchPolicy, BatchProgram, BatchRunner
from lightllm.common.execution.graph_runner import GraphRunner


def prefill_input(tokens):
    rows = int(tokens > 0)
    return ModelInput(
        batch_size=rows,
        total_token_num=tokens,
        max_q_seq_len=tokens,
        max_kv_seq_len=tokens,
        max_cache_len=0,
        input_ids=torch.arange(tokens),
        b_req_idx=torch.arange(rows, dtype=torch.int32),
        b_mtp_index=torch.zeros(rows, dtype=torch.int32),
        b_seq_len=torch.full((rows,), tokens, dtype=torch.int32),
        b_ready_cache_len=torch.zeros(rows, dtype=torch.int32),
        b_prefill_start_loc=torch.zeros(rows, dtype=torch.int32),
        b_is_decode_req=torch.zeros(rows, dtype=torch.bool),
        b_prefill_has_output_cpu=[True] * rows,
        multimodal_params=[{"images": [], "audios": []} for _ in range(rows)],
        is_prefill=True,
    )


@pytest.mark.parametrize("sizes", [(0,), (3,), (0, 3), (3, 0)])
def test_prefill_layout_and_publication_without_model(sizes):
    events, states, graph_outputs = [], [], []
    ready = SimpleNamespace(record=lambda: events.append("ready"))

    def create_state(batch, lane):
        events.append(("create", lane))
        state = SimpleNamespace(batch=batch, lane=lane, init_att_state=lambda: events.append(("attention", lane)))
        states.append(state)
        return state

    def forward(state):
        events.append(("forward", state.lane))
        batch = state.batch
        output = ModelOutput(
            logits=batch.b_req_idx[:, None],
            logits_token_ids=batch.b_req_idx.clone(),
            prompt_logics=batch.input_ids[:, None],
            mtp_collector=ModelMtpOutputCollector(spec_hidden=batch.input_ids[:, None]),
        )
        graph_outputs.append(output)
        return output

    program = BatchProgram(
        create_state=create_state,
        initialize_state=lambda state: events.append(("initialize", state.lane)),
        prefill=forward,
        decode=None,
        overlap_prefill=lambda *states: tuple(forward(state) for state in states),
        overlap_decode=None,
    )
    graphs = GraphRunner()
    graphs.prefill = SimpleNamespace(
        can_run=lambda **kwargs: True, find_closest_graph_handle_token_num=lambda **kwargs: 8
    )
    runner = BatchRunner(
        BatchPolicy(sequence_parallel_alignment=2),
        program,
        SimpleNamespace(HOLD_REQUEST_ID=88),
        graphs,
        clear_transfer=lambda: events.append("clear"),
        runtime=SimpleNamespace(create_event=lambda: ready),
    )
    inputs = tuple(prefill_input(size) for size in sizes)
    results = runner.run_prefill(*inputs)
    outputs = results if len(inputs) == 2 else (results,)
    for tokens, original, state, output, graph_output in zip(sizes, inputs, states, outputs, graph_outputs):
        padded = state.batch
        assert padded.input_ids.numel() == (8 if len(inputs) == 1 else max(2, (tokens + 1) // 2 * 2))
        assert padded.b_req_idx[-1].item() == 88
        assert padded.b_prefill_has_output_cpu[-1] is False
        assert original.batch_size == int(tokens > 0)
        assert original.input_ids.numel() == tokens
        assert original.b_prefill_has_output_cpu == [True] * original.batch_size
        torch.testing.assert_close(output.logits[:, 0], original.b_req_idx)
        torch.testing.assert_close(output.logits_token_ids, original.b_req_idx)
        torch.testing.assert_close(output.prompt_logics[:, 0], original.input_ids)
        torch.testing.assert_close(output.mtp_collector.spec_hidden[:, 0], original.input_ids)
        assert graph_output.prompt_logics.shape[0] == padded.input_ids.numel()
        assert output.prefill_mem_indexes_ready_event is ready
    if len(inputs) == 1:
        assert events == [("create", 0), "ready", ("initialize", 0), ("attention", 0), ("forward", 0)]
    else:
        assert events == [
            ("create", 0),
            ("initialize", 0),
            ("attention", 0),
            ("create", 1),
            ("initialize", 1),
            ("attention", 1),
            "ready",
            ("forward", 0),
            ("forward", 1),
            "clear",
        ]


def test_overlap_failure_does_not_release_transfer_buffers():
    def fail(*states):
        raise RuntimeError("injected transfer failure")

    runner = BatchRunner(
        BatchPolicy(),
        BatchProgram(
            create_state=lambda batch, lane: SimpleNamespace(init_att_state=lambda: None),
            initialize_state=lambda state: None,
            prefill=None,
            decode=None,
            overlap_prefill=fail,
            overlap_decode=None,
        ),
        SimpleNamespace(HOLD_REQUEST_ID=88),
        GraphRunner(),
        clear_transfer=lambda: pytest.fail("in-flight resources were released"),
        runtime=SimpleNamespace(create_event=lambda: SimpleNamespace(record=lambda: None)),
    )
    with pytest.raises(RuntimeError, match="injected transfer failure"):
        runner.run_prefill(prefill_input(1), prefill_input(1))
