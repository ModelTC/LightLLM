"""Released DSpark weights paired with the real target and an independent FP32 draft recipe."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unit_tests.common.execution.test_model_cuda import prepare_model_batch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.skipif(not os.environ.get("LIGHTLLM_TEST_DSPARK_CHECKPOINT"), reason="local DSpark checkpoint required")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.no_grad()
def test_dspark_checkpoint_target_commit_reference_and_graph(dtype, monkeypatch, record_testsuite_property):
    import gc
    import traceback

    failure = None
    with monkeypatch.context() as patch:
        try:
            _check_dspark_checkpoint(dtype, patch, record_testsuite_property)
        except Exception:
            # Keep the complete failure text without retaining a 27B model through pytest's traceback frames.
            failure = traceback.format_exc()
    gc.collect()
    for device in range(2):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    assert failure is None, failure


def _check_dspark_checkpoint(dtype, monkeypatch, record_testsuite_property):
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.execution.runner.decode_graph import CudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models import get_draft_model_class
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    directory = Path(os.environ["LIGHTLLM_TEST_DSPARK_CHECKPOINT"])
    target_dir = Path(os.environ["LIGHTLLM_TEST_DSPARK_TARGET"])
    raw = json.loads((directory / "config.json").read_text())
    cls = get_draft_model_class(raw, "dspark")
    assert raw["model_type"] == "qwen3" and raw["markov_head_type"] == "vanilla"
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
    block = 3
    args = StartArgs(
        model_dir=str(target_dir),
        data_type=dtype,
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=["fa3", "triton"],
        llm_decode_att_backend=["fa3", "triton"],
        linear_att_ssm_data_type="float32",
        mtp_mode="dspark",
        mtp_step=block,
        mtp_draft_model_dir=[str(directory)],
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    kwargs = dict(
        run_mode="normal",
        max_total_token_num=512,
        batch_max_tokens=256,
        max_req_num=2,
        max_seq_length=128,
        graph_max_batch_size=2,
        graph_max_len_in_batch=128,
        disable_cudagraph=True,
    )
    main = draft = None
    reports = []
    try:
        main = Qwen3_5TpPartModel(dict(kwargs, weight_dir=str(target_dir)))
        draft = cls(dict(kwargs, weight_dir=str(directory), main_model=main, mtp_previous_draft_models=[]))
        assert draft.model_adapter.share_main_weights
        assert draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_
        assert draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_
        assert draft._cos_cached is not main._cos_cached
        assert not draft.linear_plan.legacy
        main.prepare_attention_states()
        draft.prepare_attention_states()
        manager = main.att_state_manager
        manager.finish_state_registration()
        for state in manager.get_full_attention_states():
            state.device = torch.device("cuda:0")
            state.allocate_storage(512)
        for model in (main, draft):
            model.req_manager = SimpleNamespace(
                req_to_token_indexs=torch.arange(384, dtype=torch.int32, device="cuda:0").view(3, 128),
                HOLD_REQUEST_ID=2,
            )
            model.prefill_att_backend = model.decode_att_backend = None
            model.prefill_att_backend1 = model.decode_att_backend1 = None
            model._init_execution_plan()
            model._init_hidden_collector()
            for view in model.attention_execution.resources.stores.get("history").views().values():
                for tensor in view.fields:
                    tensor.zero_()
        monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
        monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("legacy weight compute"))

        weights = {}
        for path in directory.glob("*.safetensors"):
            with safe_open(path, "pt", "cpu") as source:
                for name in source.keys():
                    weights[name] = source.get_tensor(name).to(device="cuda:1", dtype=torch.float32)
        assert "embed_tokens.weight" not in weights and "lm_head.weight" not in weights
        head = main.pre_post_weight.lm_head_weight_.weight.to(device="cuda:1", dtype=torch.float32)
        config = Qwen3Config(**raw)
        rotary = Qwen3RotaryEmbedding(config).to("cuda:1")
        head_dim = raw["head_dim"]
        q_heads, kv_heads = raw["num_attention_heads"], raw["num_key_value_heads"]
        eps = raw["rms_norm_eps"]
        tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
        sequences = [tokenizer.encode(text) for text in ("Explain why the sky is blue.", "请简短介绍斐波那契数列。")]
        prefixes = list(map(len, sequences))
        assert max(prefixes) + 2 * block < 128
        target_input = prepare_model_batch(main, sequences, [0, 0], True, input_only=True)
        target = main.forward(target_input)
        histories = [h.clone() for h in target.mtp_collector.spec_hidden.split(prefixes)]
        anchors = target.logits.argmax(-1).tolist()
        commit = prepare_model_batch(draft, sequences, [0, 0], True, input_only=True)
        commit.mtp_draft_input_hiddens = torch.cat(histories)
        assert draft.forward(commit).logits.numel() == 0

        def linear(x, name):
            return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

        def norm(x, name):
            return F.rms_norm(x, (x.shape[-1],), weights[name + ".weight"], eps)

        def rope(x, start):
            positions = torch.arange(start, start + x.shape[0], device="cuda:1")[None]
            cos, sin = rotary(x[None], positions)
            left, right = x.chunk(2, dim=-1)
            rotated = torch.cat((-right, left), -1)
            return x * cos[0, :, None] + rotated * sin[0, :, None]

        def reference(tokens, request):
            x = F.embedding(torch.tensor(tokens, device="cuda:0"), main.pre_post_weight.wte_weight_.weight)
            x = x.to(device="cuda:1", dtype=torch.float32)
            context = norm(linear(histories[request].to(device="cuda:1", dtype=torch.float32), "fc"), "hidden_norm")
            length = prefixes[request]
            for layer in range(raw["num_hidden_layers"]):
                prefix = f"layers.{layer}."
                normalized = norm(x, prefix + "input_layernorm")
                q = norm(
                    linear(normalized, prefix + "self_attn.q_proj").view(block, q_heads, head_dim),
                    prefix + "self_attn.q_norm",
                )
                sources = torch.cat((context, normalized))
                k = norm(
                    linear(sources, prefix + "self_attn.k_proj").view(length + block, kv_heads, head_dim),
                    prefix + "self_attn.k_norm",
                )
                v = linear(sources, prefix + "self_attn.v_proj").view(length + block, kv_heads, head_dim)
                q, k = rope(q, length), rope(k, 0)
                attention = (
                    F.scaled_dot_product_attention(
                        q.transpose(0, 1),
                        k.transpose(0, 1),
                        v.transpose(0, 1),
                        enable_gqa=True,
                    )
                    .transpose(0, 1)
                    .flatten(1)
                )
                x = x + linear(attention, prefix + "self_attn.o_proj")
                normalized = norm(x, prefix + "post_attention_layernorm")
                x = x + linear(
                    F.silu(linear(normalized, prefix + "mlp.gate_proj")) * linear(normalized, prefix + "mlp.up_proj"),
                    prefix + "mlp.down_proj",
                )
            # The released DFlash backbone returns the final normalized hidden states to DSpark's heads.
            x = norm(x, "norm")
            logits = F.linear(x, head)
            previous, ids, confidence = tokens[0], [], []
            for step in range(block):
                embedding = weights["markov_head.markov_w1.weight"][previous]
                scores = logits[step] + linear(embedding, "markov_head.markov_w2")
                confidence.append(linear(torch.cat((x[step], embedding)), "confidence_head.proj"))
                previous = scores.argmax()
                ids.append(previous)
            return logits, torch.stack(ids), torch.cat(confidence)

        observed = {}
        sample = draft.post_infer._sample_markov

        def record_logits(local_logits, *args, **kwargs):
            observed["logits"] = local_logits.t().clone()
            return sample(local_logits, *args, **kwargs)

        monkeypatch.setattr(draft.post_infer, "_sample_markov", record_logits)

        def make_proposal():
            tokens = [[anchor] + [draft.mask_token_id] * (block - 1) for anchor in anchors]
            rows = [
                (r, prefixes[r] + step, token) for r, values in enumerate(tokens) for step, token in enumerate(values)
            ]
            tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda:0")
            inp = ModelInput(
                batch_size=len(rows),
                total_token_num=sum(prefixes) + len(rows),
                max_q_seq_len=1,
                max_kv_seq_len=max(prefixes) + block,
                max_cache_len=max(prefixes),
                input_ids=tensor([token for _, _, token in rows]).long(),
                b_req_idx=tensor([r for r, _, _ in rows]),
                b_seq_len=tensor([p + 1 for _, p, _ in rows]),
                b_mtp_index=tensor(list(range(block)) * 2),
                is_prefill=False,
                b_position_delta=tensor([0] * len(rows)),
                b_shared_seq_len=tensor([0] * len(rows)),
                b_shared_radix_node_id=tensor([0] * len(rows)),
                multimodal_params=[{"images": [], "audios": []}] * len(rows),
            )
            batch = draft._create_inferstate(inp)
            draft.batch_metadata(batch)
            batch.init_att_state()
            return batch, tokens

        def check(output, tokens, phase):
            expected = [reference(values, r) for r, values in enumerate(tokens)]
            logits = torch.cat([item[0] for item in expected])
            error = observed["logits"].to(device="cuda:1", dtype=torch.float32) - logits
            rms = (error.square().mean(-1) / logits.square().mean(-1)).sqrt()
            peak = error.abs().amax(-1) / logits.abs().amax(-1)
            ids = torch.stack([item[1] for item in expected])
            actual_ids = output.mtp_collector.draft_token_ids.view(2, block).to("cuda:1")
            reports.append(
                dict(
                    phase=phase, rms=rms.tolist(), peak=peak.tolist(), expected=ids.tolist(), actual=actual_ids.tolist()
                )
            )
            record_testsuite_property("dspark_" + dtype + "_" + phase, json.dumps(reports[-1]))
            assert torch.equal(actual_ids, ids), reports
            torch.testing.assert_close(
                output.mtp_collector.confidence_logits.to("cuda:1"),
                torch.stack([item[2] for item in expected]),
                rtol=2e-2,
                atol=2e-3,
            )

        batch, tokens = make_proposal()
        check(draft._token_forward(batch), tokens, "eager")
        draft.graph = CudaGraph(1, 2 * block, block, max_batch_size=2 * block, max_len_in_batch=128)
        batch.is_cuda_graph_capture = True
        captured = draft.graph.capture_decode(draft._token_forward, batch)
        check(captured, tokens, "capture")
        sequences = [values + [anchor] for values, anchor in zip(sequences, anchors)]
        target_input = prepare_model_batch(main, sequences, prefixes, False, input_only=True)
        target = main.forward(target_input)
        commit = prepare_model_batch(draft, sequences, prefixes, False, input_only=True)
        commit.mtp_draft_input_hiddens = target.mtp_collector.spec_hidden
        draft._decode(commit)
        histories = [torch.cat((h, row[None])) for h, row in zip(histories, target.mtp_collector.spec_hidden)]
        prefixes = list(map(len, sequences))
        anchors = target.logits.argmax(-1).tolist()
        batch, tokens = make_proposal()
        batch.is_cuda_graph_replay = True
        draft.graph.replay(batch)
        check(captured, tokens, "replay_after_commit")
        record_testsuite_property("dspark_" + dtype, json.dumps(reports))
        assert max(value for row in reports for value in row["rms"]) <= 0.01, reports
        assert max(value for row in reports for value in row["peak"]) <= 1 / 32, reports
    finally:
        for model in (draft, main):
            if model is not None:
                model.graph_runner.close()
        for device in range(2):
            torch.cuda.synchronize(device)
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()
