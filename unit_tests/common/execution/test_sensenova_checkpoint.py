"""Local Loop checkpoint against its independent Transformers implementation."""

from dataclasses import asdict
import gc
import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.skipif(not os.environ.get("LIGHTLLM_TEST_SENSENOVA_CHECKPOINT"), reason="local Loop checkpoint required")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.no_grad()
def test_loop_checkpoint_reference_graph_and_state_restore(monkeypatch, dtype, record_testsuite_property):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    directory = os.environ["LIGHTLLM_TEST_SENSENOVA_CHECKPOINT"]
    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    args = StartArgs(
        model_dir=directory,
        data_type=dtype,
        page_size=1,
        llm_prefill_att_backend=["triton", "triton"],
        llm_decode_att_backend=["triton", "triton"],
        linear_att_ssm_data_type="float32",
        running_max_req_size=2,
        prefill_cudagraph_max_handle_token=64,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(hf, name, None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    native = reference = None
    reports = []
    try:
        reference, loading = AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not loading["missing_keys"] and not loading.get("mismatched_keys"), loading
        assert all(key.startswith("mtp.") for key in loading["unexpected_keys"]), loading
        assert reference.config.model_type == "sensenova_loop"
        for layer in reference.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                forward = layer.linear_attn.forward
                if "use_cache" not in inspect.signature(forward).parameters:
                    # Older checkpoint code forwards this decoder flag to GDN; cache_params owns its state.
                    def without_cache_flag(*args, use_cache=None, _forward=forward, **kwargs):
                        return _forward(*args, **kwargs)

                    monkeypatch.setattr(layer.linear_attn, "forward", without_cache_flag)
        reference = reference.to("cuda:1").eval()
        torch.cuda.set_device(0)
        native = SenseNovaLoopTpPartModel(
            dict(
                run_mode="normal",
                weight_dir=directory,
                max_total_token_num=256,
                batch_max_tokens=128,
                max_req_num=2,
                max_seq_length=128,
                graph_max_batch_size=2,
                graph_max_len_in_batch=128,
                disable_cudagraph=True,
            )
        )
        native.prepare_attention_states()
        native.att_state_manager.finish_state_registration()
        for state in native.att_state_manager.get_full_attention_states():
            state.device = torch.device("cuda")
            state.allocate_storage(384)
        native.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(384, dtype=torch.int32, device="cuda").view(3, 128),
            HOLD_REQUEST_ID=2,
        )
        native.prefill_att_backend = native.decode_att_backend = None
        native.prefill_att_backend1 = native.decode_att_backend1 = None
        native._init_execution_plan()
        native._init_hidden_collector()
        assert native.execution_plan and native.linear_plan.bindings and not native.linear_plan.legacy
        monkeypatch.setattr(
            native.att_state_manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup")
        )

        indices = native.loop_layer_indices
        assert len(native.physical_layer_weights) == len(set(indices)) < len(native.layers_infer)
        views = {
            **native.attention_execution.resources.stores.get("history").views(),
            **native.attention_execution.resources.stores.get("recurrent").views(),
        }
        by_layer = {view.state.producer: view for view in views.values()}
        for physical in set(indices):
            visits = [i for i, p in enumerate(indices) if p == physical]
            assert all(native.trans_layers_weight[i] is native.trans_layers_weight[visits[0]] for i in visits)
            for field in (0, 1):
                assert len({by_layer[i].fields[field].data_ptr() for i in visits}) == len(visits)

        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
        sequences = [tokenizer.encode(s) for s in ("请简短解释为什么天空是蓝色的。", "用 Python 写一个计算斐波那契数列的函数。")]
        assert max(map(len, sequences)) < 64

        def check(output, stage):
            target = torch.cat(
                [
                    reference(torch.tensor([seq], device="cuda:1"), use_cache=False).logits[:, -1].float().cpu()
                    for seq in sequences
                ]
            )
            actual = output.logits.float().cpu()
            error = actual - target
            reports.append(
                dict(
                    stage=stage,
                    rms=(error.square().mean(-1) / target.square().mean(-1)).sqrt().tolist(),
                    peak=(error.abs().amax(-1) / target.abs().amax(-1)).tolist(),
                    greedy=actual.argmax(-1).tolist(),
                    expected=target.argmax(-1).tolist(),
                )
            )
            return target.argmax(-1).tolist()

        def snapshot():
            return [(tensor, tensor.clone()) for view in views.values() for tensor in view.fields[:2]]

        def restore(saved):
            for tensor, value in saved:
                tensor.copy_(value)

        batch = prepare_model_batch(native, sequences, [0, 0], True)
        tokens = check(native._context_forward(batch), "prefill")
        for step in range(2):
            cached = list(map(len, sequences))
            sequences = [seq + [token] for seq, token in zip(sequences, tokens)]
            batch = prepare_model_batch(native, sequences, cached, False)
            tokens = check(native._token_forward(batch), f"decode{step}")

        saved = snapshot()
        native.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=128)
        batch.is_cuda_graph_capture = True
        captured = native.graph.capture_decode(native._token_forward, batch)
        restore(saved)
        cached = list(map(len, sequences))
        sequences = [seq + [token] for seq, token in zip(sequences, tokens)]
        batch = prepare_model_batch(native, sequences, cached, False)
        batch.is_cuda_graph_replay = True
        native.graph.replay(batch)
        check(captured, "decode_graph")

        cached = list(map(len, sequences))
        suffix = tokenizer.encode("请再简短一点。")
        sequences = [seq + suffix for seq in sequences]
        batch = prepare_model_batch(native, sequences, cached, True)
        saved = snapshot()
        check(native._context_forward(batch), "suffix")
        restore(saved)
        native.prefill_graph = PrefillCudaGraph(native.graph, 1)
        batch = prepare_model_batch(native, sequences, cached, True, input_only=True)
        tokens = check(native.forward(batch), "prefill_graph")
        assert native.prefill_graph.graph

        for state in native.att_state_manager.get_bounded_attention_states():
            state.initialize_page_pools(2, device="cuda", pin_memory=False)
            state.save_bounded_att_runtime_to_pages([0, 1], list(map(len, sequences)), "big", [0, 1])
            state.conv_state.zero_()
            state.ssm_state.zero_()
            state.restore_bounded_att_runtime_from_pages([0, 1], list(map(len, sequences)), "big", [0, 1])
        cached = list(map(len, sequences))
        sequences = [seq + [token] for seq, token in zip(sequences, tokens)]
        batch = prepare_model_batch(native, sequences, cached, False)
        tokens = check(native._token_forward(batch), "restored_decode")
        assert all(row["greedy"] == row["expected"] for row in reports), reports
        assert max(value for row in reports for value in row["rms"]) <= 0.01, reports
        assert max(value for row in reports for value in row["peak"]) <= 1 / 32, reports
    finally:
        record_testsuite_property(f"loop_comparison_{dtype}", json.dumps(reports))
        if native is not None:
            torch.cuda.synchronize(0)
            native.graph_runner.close()
        if reference is not None:
            torch.cuda.synchronize(1)
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
        native = reference = None
        gc.collect()
