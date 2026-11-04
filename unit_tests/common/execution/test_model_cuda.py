"""Full tiny-model forwards against independent Transformers reference weights."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3", "qwen3_window", "llama_wide", "qwen3_wide"])
@pytest.mark.parametrize(
    "prefill_implementation,decode_implementation,kv_type",
    [
        ("triton", "triton", "None"),
        ("fa3", "fa3", "None"),
        ("triton", "triton", "int4kv"),
        ("triton", "triton", "int8kv"),
        ("fa3", "fa3", "fp8kv_sph"),
        ("fa3", "fa3", "fp8kv_spt"),
        ("flashinfer", "flashinfer", "None"),
        ("flashinfer", "flashinfer", "fp8kv_spt"),
        ("fa3", "flashinfer", "None"),
        ("flashinfer", "fa3", "fp8kv_spt"),
    ],
    ids=[
        "triton",
        "fa3",
        "int4",
        "int8",
        "fp8_head",
        "fp8_tensor",
        "flashinfer",
        "flashinfer_fp8",
        "mixed",
        "mixed_fp8",
    ],
)
@torch.no_grad()
def test_tiny_model_prefill_decode_prefix_and_graph(
    family,
    prefill_implementation,
    decode_implementation,
    kv_type,
    monkeypatch,
    tmp_path,
    gdn_implementation="triton",
    fused_shared_experts=False,
    startup=False,
    model_dtype=None,
    verify_layout=None,
    ssm_dtype="float32",
    gptoss_packed=False,
    gptoss_yarn=False,
    gptoss_topk=2,
    mhc_pre_scale=1.0,
    mhc_fixed_identity=False,
):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from lightllm.models.llama.model import LlamaTpPartModel
    from lightllm.models.qwen2.model import Qwen2TpPartModel
    from lightllm.models.qwen3.model import Qwen3TpPartModel
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from lightllm.models.qwen3_moe.model import Qwen3MOEModel
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.models.sensenova_loop.model import SenseNovaLoopTpPartModel
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM
    from lightllm.models.deepseek2.model import Deepseek2TpPartModel
    from lightllm.models.glm4_moe_lite.model import Glm4MoeLiteTpPartModel
    from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
    from transformers import GptOssConfig, GptOssForCausalLM
    from lightllm.models.gpt_oss.model import GptOssTpPartModel
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type

    model_types = {
        "llama": (LlamaConfig, LlamaForCausalLM, LlamaTpPartModel),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM, Qwen2TpPartModel),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM, Qwen3TpPartModel),
        "gpt_oss": (GptOssConfig, GptOssForCausalLM, GptOssTpPartModel),
        "qwen3_window": (Qwen3Config, Qwen3ForCausalLM, Qwen3TpPartModel),
        "llama_wide": (LlamaConfig, LlamaForCausalLM, LlamaTpPartModel),
        "qwen3_wide": (Qwen3Config, Qwen3ForCausalLM, Qwen3TpPartModel),
        "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MOEModel),
        "qwen3_moe_dense": (Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MOEModel),
        "sensenova_loop": (qwen35.Qwen3_5TextConfig, qwen35.Qwen3_5ForCausalLM, SenseNovaLoopTpPartModel),
        "sensenova_mhc2": (qwen35.Qwen3_5TextConfig, qwen35.Qwen3_5ForCausalLM, SenseNovaLoopTpPartModel),
        "sensenova_mhc4": (qwen35.Qwen3_5TextConfig, qwen35.Qwen3_5ForCausalLM, SenseNovaLoopTpPartModel),
        "qwen3next": (Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextTpPartModel),
        "qwen3next_moe": (Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextTpPartModel),
        "qwen3next_irregular": (Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextTpPartModel),
        "qwen35": (qwen35.Qwen3_5TextConfig, qwen35.Qwen3_5ForCausalLM, Qwen3_5TpPartModel),
        "deepseek2": (DeepseekV2Config, DeepseekV2ForCausalLM, Deepseek2TpPartModel),
        "deepseek2_lora": (DeepseekV2Config, DeepseekV2ForCausalLM, Deepseek2TpPartModel),
        "deepseek3_yarn": (DeepseekV3Config, DeepseekV3ForCausalLM, Deepseek2TpPartModel),
        "deepseek3_moe": (DeepseekV3Config, DeepseekV3ForCausalLM, Deepseek2TpPartModel),
        "glm4_moe_lite": (DeepseekV3Config, DeepseekV3ForCausalLM, Glm4MoeLiteTpPartModel),
        "deepseek3_2": (DeepseekV3Config, DeepseekV3ForCausalLM, Deepseek3_2TpPartModel),
        "deepseek3_2_flashmla": (DeepseekV3Config, DeepseekV3ForCausalLM, Deepseek3_2TpPartModel),
    }
    qwen35_reference = qwen35
    if family == "qwen35_moe":
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_reference
        from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel

        model_types[family] = (
            qwen35_reference.Qwen3_5MoeTextConfig,
            qwen35_reference.Qwen3_5MoeForCausalLM,
            Qwen3_5MOETpPartModel,
        )
    config_cls, reference_cls, model_cls = model_types[family]
    sparse = family.startswith("deepseek3_2")
    dtype = model_dtype or (torch.bfloat16 if sparse else torch.float16)
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
        model_dir=str(tmp_path),
        data_type="bfloat16" if dtype == torch.bfloat16 else "float16",
        page_size=1,
        llm_kv_type=kv_type,
        llm_kv_quant_group_size=32,
        kv_quant_calibration_config_path=str(tmp_path / "calibration.json")
        if kv_type.startswith("fp8") and not sparse
        else None,
        llm_prefill_att_backend=[prefill_implementation, gdn_implementation],
        llm_decode_att_backend=[decode_implementation, "triton"],
        linear_att_ssm_data_type=ssm_dtype,
        enable_fused_shared_experts=fused_shared_experts,
        running_max_req_size=2,
        prefill_cudagraph_max_handle_token=5,
        mtp_mode="eagle_with_att" if verify_layout else None,
        mtp_step=2 if verify_layout else 0,
        mtp_dynamic_verify=verify_layout == "dynamic",
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    torch.manual_seed(41)
    loop = family.startswith("sensenova")
    recurrent = family.startswith(("qwen3next", "qwen35")) or loop
    mla = family.startswith(("deepseek", "glm4_moe_lite"))
    extra = (
        dict(
            num_experts=0,
            full_attention_interval=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        if recurrent
        else {}
    )
    if family in ("qwen3next_moe", "qwen35_moe"):
        extra.update(
            num_experts=4,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
    if family.startswith("qwen3_moe"):
        extra.update(num_experts=4, moe_intermediate_size=256, num_experts_per_tok=2, norm_topk_prob=True)
        if family.endswith("_dense"):
            extra.update(mlp_only_layers=[0])
    if family == "gpt_oss":
        extra.update(
            num_local_experts=4,
            num_experts_per_tok=gptoss_topk,
            sliding_window=3,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0),
        )
        if gptoss_yarn:
            extra["rope_parameters"] = dict(
                rope_type="yarn",
                rope_theta=150000.0,
                factor=4.0,
                beta_fast=4.0,
                beta_slow=1.0,
                original_max_position_embeddings=8,
                truncate=False,
            )
    if loop or family.startswith("qwen35"):
        extra.update(
            layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0, mrope_section=[11, 11, 10]
            ),
        )
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ):
            monkeypatch.setattr(qwen35_reference, name, None)
    if mla:
        extra = dict(
            n_routed_experts=4,
            n_shared_experts=1,
            first_k_dense_replace=2,
            num_experts_per_tok=2,
            q_lora_rank=None if family == "deepseek2" else 64,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_scaling=(
                {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 32}
                if family.endswith("_lora")
                else None
            ),
        )
        if family == "deepseek3_yarn":
            extra["rope_scaling"] = dict(
                rope_type="yarn", factor=2.0, mscale=1, mscale_all_dim=1, original_max_position_embeddings=32
            )
        if family in ("deepseek3_moe", "glm4_moe_lite"):
            extra.update(
                first_k_dense_replace=1,
                n_routed_experts=4,
                n_shared_experts=1,
                moe_intermediate_size=256,
                n_group=2,
                topk_group=1,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                routed_scaling_factor=1.5,
                scoring_func="sigmoid",
            )
    config = config_cls(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4 if recurrent else 2,
        num_attention_heads=64 if family == "deepseek3_2_flashmla" else 4,
        num_key_value_heads=64 if family == "deepseek3_2_flashmla" else 4 if mla else 2,
        head_dim=128 if family.endswith("_wide") else 64,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        **extra,
    )
    if family == "qwen3next_irregular":
        config.layer_types = ["full_attention", "linear_attention", "full_attention", "linear_attention"]
    if family == "qwen3_window":
        config.sliding_window = 3
        config.use_sliding_window = True
        config.layer_types = ["full_attention", "sliding_attention"]
    reference = reference_cls(config).to(dtype=dtype, device="cuda").eval()
    reference.config._attn_implementation = "eager"
    if sparse:
        from types import MethodType

        for layer in reference.model.layers:
            layer.self_attn.forward = MethodType(absorbed_mla_reference, layer.self_attn)
    config.architectures = [reference_cls.__name__]
    config.save_pretrained(tmp_path)
    if family == "glm4_moe_lite":
        native_config = config.to_dict()
        native_config.update(model_type="glm4_moe_lite", architectures=["Glm4MoeLiteForCausalLM"])
        (tmp_path / "config.json").write_text(json.dumps(native_config))
    native_weights = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    if family == "gpt_oss":
        from unit_tests.common.execution.test_gpt_oss import gpt_oss_checkpoint

        native_weights = gpt_oss_checkpoint(reference, gptoss_packed)
    if loop:
        from types import MethodType
        from lightllm.models.sensenova_loop.config import get_loop_layer_indices

        settings = dict(start=1, end=3, eval_count=2, use_gate=False)
        indices = get_loop_layer_indices(settings, config.num_hidden_layers)
        native_config = dict(model_type="sensenova_loop", text_config=config.to_dict(), loop_settings=settings)
        physical = list(reference.model.layers)
        if family != "sensenova_loop":
            streams = int(family[-1])
            mhc = dict(start=1, end=3, streams=streams, iterations=20)
            mhc.update(pre_scale=mhc_pre_scale, fixed_identity=mhc_fixed_identity)
            native_config["mhc"] = mhc
            for i in range(1, 3):
                layer = physical[i]
                layer.loop_mhc = mhc
                layer.loop_entry, layer.loop_exit = i == 1, i == 2
                for branch in ("attn", "ffn"):
                    rows = 2 * streams + (0 if mhc_fixed_identity else streams * streams)
                    weight = torch.randn(rows, streams * config.hidden_size, device="cuda") * 0.005
                    bias = torch.randn(rows, device="cuda") * 0.1
                    setattr(layer, f"loop_{branch}", (weight, bias))
                    native_weights[f"model.layers.{i}.mhc_{branch}.weight"] = weight.cpu()
                    native_weights[f"model.layers.{i}.mhc_{branch}.bias"] = bias.cpu()
                layer.forward = MethodType(loop_mhc_reference, layer)
        (tmp_path / "config.json").write_text(json.dumps(native_config))
        reference.model.layers = torch.nn.ModuleList([physical[i] for i in indices])
        config.layer_types = [config.layer_types[i] for i in indices]
        config.num_hidden_layers = len(indices)
    # New Transformers versions fuse expert parameters internally. Export the
    # checkpoint's individual expert tensors for the model adapter.
    for name in () if family in ("gpt_oss", "qwen35_moe") else tuple(native_weights):
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for index, weight in enumerate(native_weights.pop(name)):
                gate, up = weight.chunk(2, dim=0)
                native_weights[f"{prefix}{index}.gate_proj.weight"] = gate
                native_weights[f"{prefix}{index}.up_proj.weight"] = up
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for index, weight in enumerate(native_weights.pop(name)):
                native_weights[f"{prefix}{index}.down_proj.weight"] = weight
    if sparse:
        native_config = config.to_dict()
        native_config.update(
            model_type="deepseek_v32",
            architectures=["DeepseekV32ForCausalLM"],
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
        )
        (tmp_path / "config.json").write_text(json.dumps(native_config))
        for i in range(2):
            prefix = f"model.layers.{i}.self_attn.indexer"
            for name, shape in (
                ("wq_b.weight", (64 * 128, 64)),
                ("wk.weight", (128, 256)),
                ("weights_proj.weight", (64, 256)),
            ):
                native_weights[f"{prefix}.{name}"] = (torch.randn(shape) * 0.02).to(dtype)
            native_weights[f"{prefix}.k_norm.weight"] = torch.ones(128, dtype=dtype)
            native_weights[f"{prefix}.k_norm.bias"] = torch.zeros(128, dtype=dtype)
    if kv_type.startswith("fp8") and not sparse:
        width = 4 if kv_type == "fp8kv_sph" else 2
        calibration = dict(
            architectures=reference_cls.__name__,
            num_layers=2,
            num_head=2,
            scales_shape=[2, width],
            scales=[[0.0078125] * width] * 2,
            quant_type="per_head" if kv_type == "fp8kv_sph" else "per_tensor",
        )
        (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    # The adapter owns construction; old model-class factories must be unused.
    for name in (
        "pre_and_post_weight_class",
        "transformer_weight_class",
        "pre_layer_infer_class",
        "post_layer_infer_class",
        "transformer_layer_infer_class",
    ):
        monkeypatch.setattr(model_cls, name, lambda *a, **kw: pytest.fail("legacy model factory"))
    model = model_cls(
        dict(
            run_mode="normal",
            weight_dir=str(tmp_path),
            max_total_token_num=64,
            batch_max_tokens=16,
            max_req_num=2,
            max_seq_length=16,
            graph_max_batch_size=2,
            graph_max_len_in_batch=16,
            disable_cudagraph=True,
            weight_dict=native_weights,
        )
    )
    assert model.linear_plan.bindings and not model.linear_plan.legacy
    if loop:
        assert len(model.physical_layer_weights) == 4 and len(model.trans_layers_weight) == 6
        for i, physical_id in enumerate(indices):
            assert model.trans_layers_weight[i] is model.physical_layer_weights[physical_id]
    assert len(model.moe_plan.bindings) == len(model.model_adapter.moe_specs())
    assert not model.moe_plan.legacy
    if family == "qwen35_moe":
        assert len(model.moe_plan.bindings) == config.num_hidden_layers
    for layer_id, bound in model.moe_plan.bindings:
        assert model.layers_infer[layer_id].bound_moe is bound
        monkeypatch.setattr(
            model.trans_layers_weight[layer_id].experts,
            "experts",
            lambda *a, **kw: pytest.fail("runtime weight-loader MoE"),
            raising=False,
        )
    from lightllm.common.quantization.no_quant import NoQuantization

    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("runtime weight-loader compute"))
    model.prepare_attention_states()
    for spec in model.model_adapter.attention_specs().values():
        if hasattr(spec, "head_dim"):
            assert spec.head_dim == config.head_dim
    assert model.execution_runner is not None
    assert model.execution_runner.prepared.workspace_bytes > 0
    if kv_type.startswith("fp8") and not sparse:
        from lightllm.core.formats.kv.calibration import calibration_table_identity

        expected_identity = calibration_table_identity(calibration, calibration["quant_type"])
        assert model.execution_runner.store.format.calibration == expected_identity
    model.att_state_manager.finish_state_registration()
    model.att_state_manager.load_full_att_calibration(args)
    for state in model.att_state_manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(64)
    model._init_execution_plan()
    if loop:
        views = {**model.execution_runner.store.views(), **model.execution_runner.recurrent_store.views()}
        by_layer = {view.state.producer: view for view in views.values()}
        for a, b in ((1, 3), (2, 4)):
            assert by_layer[a].state.key != by_layer[b].state.key
            for first, second in zip(by_layer[a].fields[:2], by_layer[b].fields[:2]):
                assert first.data_ptr() != second.data_ptr()
    model._init_hidden_collector()
    model.prefill_att_backend = model.decode_att_backend = None
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    model.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.arange(32, dtype=torch.int32, device="cuda").view(2, 16)
    )
    startup_allocations = []
    if startup:
        from unit_tests.common.execution.test_startup_runner import make_resources

        model.req_manager, allocator = make_resources(monkeypatch, device="cuda", page_size=1)
        model.att_state_manager.full_att_allocator = allocator
        # This is a single-process model test; device operations are real, without a process-group barrier.
        model.warmup_runner.runtime = SimpleNamespace(
            synchronize=torch.cuda.synchronize, empty_cache=torch.cuda.empty_cache, barrier=lambda: None
        )
    # Any resource lookup after binding is a regression.
    monkeypatch.setattr(model.att_state_manager, "get_state", lambda **_: pytest.fail("runtime resource lookup"))

    def prepare(sequences, cached, prefill):
        return prepare_model_batch(model, sequences, cached, prefill)

    def expected(sequences):
        outputs = []
        for sequence in sequences:
            kwargs = {}
            if loop:
                kwargs["use_cache"] = False
            if sparse:
                # Short sequences select all keys. Packed FP8 is compared to the
                # unquantized reference; decoded kernel values are checked separately.
                pass
            elif kv_type.startswith("fp8"):
                kwargs["past_key_values"] = FP8ReferenceCache()
            elif kv_type != "None":
                kwargs["past_key_values"] = GroupQuantizedReferenceCache(4 if kv_type == "int4kv" else 8, 32)
            outputs.append(reference(torch.tensor([sequence], device="cuda"), **kwargs).logits[:, -1].float())
        return torch.cat(outputs)

    def check(output, sequences):
        target = expected(sequences)
        if dtype == torch.bfloat16 and kv_type == "None":
            # BF16 projection/normalization rounding differs from the FP16 matrix above.
            error = output.logits.float() - target
            assert (error.square().mean() / target.square().mean()).sqrt().item() < 0.015
            assert error.abs().max() < target.abs().max() / 64
            assert torch.equal(output.logits.argmax(-1), target.argmax(-1))
        elif kv_type == "None":
            torch.testing.assert_close(output.logits, target, rtol=1e-2, atol=4e-3)
        else:
            # INT truncation amplifies small RoPE differences. FA3 FP8 additionally
            # quantizes Q and computes attention at lower precision than this HF reference.
            # Encoding and decoded-attention checks are covered separately.
            error = output.logits.float() - target
            relative_rms = (error.square().mean() / target.square().mean()).sqrt().item()
            low_precision = kv_type == "int4kv" or kv_type.startswith("fp8")
            assert relative_rms < (0.05 if low_precision else 0.01), relative_rms
            assert error.abs().max().item() < (0.075 if low_precision else 0.015)

    def snapshot():
        if model.execution_runner.recurrent_store is None:
            return []
        views = list(model.execution_runner.store.views().values())
        views += list(model.execution_runner.recurrent_store.views().values())
        return [(tensor, tensor.clone()) for view in views for tensor in view.fields[:2]]

    def restore(saved):
        for tensor, value in saved:
            tensor.copy_(value)

    try:
        if verify_layout:
            exercise_verification(model, prepare, check, verify_layout)
            return
        if startup:
            live = model.req_manager.alloc()
            indices = allocator.alloc(16).clone()
            startup_allocations.append((live, indices))
            model.req_manager.req_to_token_indexs[live, :16] = indices.cuda()
            mapping = model.req_manager.req_to_token_indexs[live].clone()
            from lightllm.core.runner import warmup as warmup_module

            with monkeypatch.context() as warmup_patch:
                warmup_patch.setattr(warmup_module.logger, "exception", lambda *args: pytest.fail(str(args)))
                model._autotune_warmup()
            model._init_padded_req()
            model.disable_cudagraph = False
            model.args.enable_prefill_cudagraph = True
            model._init_cudagraph()
            model._init_prefill_cuda_graph()
            model._check_max_len_infer()
            assert model.req_manager.req_list.marks == [1, 0]
            assert allocator.can_use_mem_size == 48
            torch.testing.assert_close(model.req_manager.req_to_token_indexs[live], mapping)
            assert set(model.graph.graph) == {1, 2}
            assert set(model.prefill_graph.graph) == {4, 5}
            second = model.req_manager.alloc()
            indices = allocator.alloc(16).clone()
            startup_allocations.append((second, indices))
            model.req_manager.req_to_token_indexs[second, :16] = indices.cuda()

        sequences = [[11, 12, 13], [21, 22]]
        if startup:
            check(model.forward(prepare_model_batch(model, sequences, [0, 0], True, input_only=True)), sequences)
        else:
            batch = prepare(sequences, [0, 0], True)
            check(model._context_forward(batch), sequences)
        sequences = [[11, 12, 13, 14], [21, 22, 23]]
        batch = prepare(sequences, [3, 2], False)
        if startup:
            check(model.forward(prepare_model_batch(model, sequences, [3, 2], False, input_only=True)), sequences)
        else:
            check(model._token_forward(batch), sequences)
        saved = snapshot()
        model.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=16)
        batch.is_cuda_graph_capture = True
        captured = model.graph.capture_decode(model._token_forward, batch)
        # Startup capture uses dummy requests; restore live state in this direct capture test.
        restore(saved)
        sequences = [[11, 12, 13, 14, 15], [21, 22, 23, 24]]
        new_batch = prepare(sequences, [4, 3], False)
        new_batch.is_cuda_graph_replay = True
        model.graph.replay(new_batch)
        check(captured, sequences)
        sequences = [[11, 12, 13, 14, 15, 16, 17], [21, 22, 23, 24, 25, 26]]
        batch = prepare(sequences, [5, 4], True)
        saved = snapshot()
        check(model._context_forward(batch), sequences)
        restore(saved)
        model.prefill_graph = PrefillCudaGraph(model.graph, 1)
        check(model._context_forward(batch), sequences)
        sequences = [[11, 12, 13, 14, 15, 16, 17, 18, 19], [21, 22, 23, 24, 25, 26, 27, 28]]
        batch = prepare(sequences, [7, 6], True)
        check(model._context_forward(batch), sequences)
    finally:
        torch.cuda.synchronize()
        model.graph_runner.close()
        if startup_allocations:
            torch.cuda.synchronize()
            for request_id, indices in startup_allocations:
                allocator.free(indices)
                model.req_manager.free_req(request_id)
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()


def exercise_verification(model, prepare, check, layout):
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.core.runner.prefill_graph import PrefillCudaGraph

    assert {item.phase for item in model.execution_plan.bindings} == {"prefill", "verify"}
    assert model.execution_runner.prepared.selections[0].offer.execution_limits.max_batch_size >= 9
    model.req_manager.req_to_token_indexs = torch.arange(48, dtype=torch.int32, device="cuda").view(3, 16)
    model.req_manager.HOLD_REQUEST_ID = 2
    prompts = [[11, 12, 13], [21, 22]]
    check(model._context_forward(prepare(prompts, [0, 0], True)), prompts)
    store = model.execution_runner.recurrent_store
    states = list({id(state): state for _, state in store.resources.values()}.values()) if store else []

    def snapshot():
        tensors = [tensor for view in model.execution_runner.store.views().values() for tensor in view.fields]
        tensors += [t for state in states for t in (state.conv_state, state.ssm_state, state.mtp_state_index)]
        if model.execution_runner.window_store is not None:
            tensors += [t for view in model.execution_runner.window_store.views().values() for t in view.fields]
        return [(tensor, tensor.clone()) for tensor in tensors] if states else []

    def restore(saved):
        for tensor, value in saved:
            tensor.copy_(value)

    def commit(accepted):
        model.att_state_manager.commit_bounded_att_state(
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            torch.tensor(accepted, device="cuda", dtype=torch.int32),
            role="main",
        )

    def batch_for(prefixes, proposals):
        sequences, ids, requests, lengths, steps = [], [], [], [], []
        for request, (prefix, tokens) in enumerate(zip(prefixes, proposals)):
            for step, token in enumerate(tokens):
                sequences.append(prefix + tokens[: step + 1])
                ids.append(token)
                requests.append(request)
                lengths.append(len(sequences[-1]))
                steps.append(step)
        padding = 6 - len(ids)
        ids += [1] * padding
        requests += [2] * padding
        lengths += [2] * padding
        steps += [0] * padding
        tensor = lambda data: torch.tensor(data, dtype=torch.int32, device="cuda")
        inp = ModelInput(
            batch_size=6,
            total_token_num=sum(lengths),
            max_q_seq_len=1,
            max_kv_seq_len=max(lengths),
            max_cache_len=max(lengths) - 1,
            input_ids=tensor(ids).long(),
            b_req_idx=tensor(requests),
            b_seq_len=tensor(lengths),
            b_mtp_index=tensor(steps),
            is_prefill=False,
            b_position_delta=tensor([0] * 6),
            b_shared_seq_len=tensor([0] * 6),
            b_shared_radix_node_id=tensor([0] * 6),
            multimodal_params=[{"images": [], "audios": []}] * 6,
        )
        state = model._create_inferstate(inp)
        model.batch_metadata(state)
        state.init_att_state()
        return state, sequences

    proposals = [[14, 15, 16], [23] if layout == "dynamic" else [23, 24, 25]]
    batch, sequences = batch_for(prompts, proposals)
    output = model._token_forward(batch)
    assert output.mtp_collector.spec_hidden.shape[0] == 6
    check(SimpleNamespace(logits=output.logits[: len(sequences)]), sequences)
    saved = snapshot()
    model.graph = CudaGraph(1, 6, 1, max_batch_size=6, max_len_in_batch=16)
    batch.is_cuda_graph_capture = True
    captured = model.graph.capture_decode(model._token_forward, batch)
    restore(saved)

    # Keep different accepted prefixes, then overwrite the rejected speculative suffix.
    prefixes = [prompts[0] + [14], prompts[1] + proposals[1][:2]]
    if states:
        commit([1, min(2, len(proposals[1]))])
    proposals = [[31] if layout == "dynamic" else [31, 32, 33], [41, 42, 43]]
    batch, sequences = batch_for(prefixes, proposals)
    batch.is_cuda_graph_replay = True
    model.graph.replay(batch)
    check(SimpleNamespace(logits=captured.logits[: len(sequences)]), sequences)
    if states:
        commit([1, 2])
        prefixes = [prefix + tokens[:n] for prefix, tokens, n in zip(prefixes, proposals, [1, 2])]
        for state in states:
            state.initialize_page_pools(2, device="cuda", pin_memory=False)
            state.save_bounded_att_runtime_to_pages([0, 1], list(map(len, prefixes)), "big", [0, 1])
        saved = snapshot()

    # A new suffix prefill must observe only the accepted prefix, even though
    # later physical slots still contain rejected KV values.
    continued = [prefixes[0] + [51, 52], prefixes[1] + [61, 62]]
    check(model._context_forward(prepare(continued, list(map(len, prefixes)), True)), continued)
    if states:
        restore(saved)
        model.prefill_graph = PrefillCudaGraph(model.graph, 1)
        check(model._context_forward(prepare(continued, list(map(len, prefixes)), True)), continued)
        restore(saved)
        for state in states:
            state.reset_bounded_att_state([0, 1])
            state.restore_bounded_att_runtime_from_pages([0, 1], list(map(len, prefixes)), "big", [0, 1])
            assert state.mtp_state_index.tolist() == [0, 0, 0]
        check(model._context_forward(prepare(continued, list(map(len, prefixes)), True)), continued)


@pytest.mark.parametrize(
    "family", ["qwen3next", "qwen3next_irregular", "qwen3next_moe", "qwen35", "sensenova_loop", "sensenova_mhc2"]
)
@pytest.mark.parametrize("dense,gdn", [("triton", "triton"), ("fa3", "triton"), ("flashinfer", "flashqla")])
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_tiny_recurrent_verification(family, dense, gdn, layout, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family, dense, dense, "None", monkeypatch, tmp_path, gdn, verify_layout=layout
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ssm_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("gdn", ["triton", "flashqla"])
def test_tiny_recurrent_verification_precision(dtype, ssm_dtype, gdn, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        "qwen3next",
        "triton",
        "triton",
        "None",
        monkeypatch,
        tmp_path,
        gdn,
        model_dtype=dtype,
        verify_layout="dynamic",
        ssm_dtype=ssm_dtype,
    )


@pytest.mark.parametrize("family", ["llama", "qwen3", "qwen3_window", "qwen3_moe"])
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_tiny_model_speculative_verification(family, backend, layout, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family,
        backend,
        backend,
        "None",
        monkeypatch,
        tmp_path,
        verify_layout=layout,
    )


@pytest.mark.parametrize(
    "backend,kv_type",
    [
        ("triton", "int4kv"),
        ("triton", "int8kv"),
        ("fa3", "fp8kv_sph"),
        ("fa3", "fp8kv_spt"),
        ("flashinfer", "fp8kv_spt"),
    ],
)
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_tiny_model_quantized_verification(backend, kv_type, layout, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        "qwen3",
        backend,
        backend,
        kv_type,
        monkeypatch,
        tmp_path,
        verify_layout=layout,
    )


@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
def test_tiny_model_bf16_verification(backend, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        "qwen3",
        backend,
        backend,
        "None",
        monkeypatch,
        tmp_path,
        model_dtype=torch.bfloat16,
        verify_layout="dynamic",
    )


@pytest.mark.parametrize(
    "family,implementation",
    [("llama", "triton"), ("llama", "flashinfer"), ("qwen3next", "triton"), ("deepseek2", "triton")],
)
def test_tiny_model_startup_preserves_allocations_and_replays(family, implementation, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family, implementation, implementation, "None", monkeypatch, tmp_path, startup=True
    )


@pytest.mark.parametrize("family", ["qwen3next", "qwen3next_irregular", "qwen3next_moe"])
@pytest.mark.parametrize("dense,gdn", [("triton", "triton"), ("fa3", "triton"), ("flashinfer", "flashqla")])
def test_tiny_recurrent_model(family, dense, gdn, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(family, dense, dense, "None", monkeypatch, tmp_path, gdn)


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_moe_dense"])
@pytest.mark.parametrize("backend", ["triton", "fa3", "flashinfer"])
def test_tiny_qwen3_moe_model(family, backend, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(family, backend, backend, "None", monkeypatch, tmp_path)


@pytest.mark.parametrize("family", ["sensenova_loop", "sensenova_mhc2", "sensenova_mhc4"])
@pytest.mark.parametrize("dense,gdn", [("triton", "triton"), ("fa3", "triton"), ("flashinfer", "flashqla")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tiny_loop_model(family, dense, gdn, dtype, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family, dense, dense, "None", monkeypatch, tmp_path, gdn, model_dtype=dtype
    )


@pytest.mark.parametrize("family", ["sensenova_mhc2", "sensenova_mhc4"])
@pytest.mark.parametrize("pre_scale,fixed_identity", [(0.25, False), (1.0, True), (3.0, True)])
def test_tiny_loop_model_mhc_options(family, pre_scale, fixed_identity, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family,
        "triton",
        "triton",
        "None",
        monkeypatch,
        tmp_path,
        model_dtype=torch.float16,
        mhc_pre_scale=pre_scale,
        mhc_fixed_identity=fixed_identity,
    )


@pytest.mark.parametrize("dense,gdn", [("triton", "triton"), ("fa3", "triton"), ("flashinfer", "flashqla")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tiny_qwen35_moe_model(dense, gdn, dtype, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        "qwen35_moe", dense, dense, "None", monkeypatch, tmp_path, gdn, model_dtype=dtype
    )


@pytest.mark.parametrize("dense,gdn", [("triton", "triton"), ("fa3", "triton"), ("flashinfer", "flashqla")])
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_tiny_qwen35_moe_verification(dense, gdn, layout, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        "qwen35_moe", dense, dense, "None", monkeypatch, tmp_path, gdn, verify_layout=layout
    )


def loop_mhc_reference(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, **kwargs):
    """Independent Torch routing around the reference model's attention and MLP."""
    streams, iterations = self.loop_mhc["streams"], self.loop_mhc["iterations"]
    batch, length = hidden_states.shape[:2]
    if self.loop_entry:
        hidden_states = hidden_states.unsqueeze(-2).expand(-1, -1, streams, -1).flatten(-2)
    residual = hidden_states.reshape(batch * length, streams, -1)

    def route(x, parameters, norm):
        flat = x.flatten(1).float()
        weight, bias = parameters
        logits = torch.nn.functional.linear(
            flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6), weight, bias
        )
        pre = logits[:, :streams].sigmoid().to(x.dtype)
        post = (2 * logits[:, streams : 2 * streams].sigmoid()).to(x.dtype)
        if self.loop_mhc.get("fixed_identity", False):
            mix = torch.eye(streams, device=x.device).expand(x.shape[0], -1, -1)
        else:
            mix = logits[:, 2 * streams :].reshape(-1, streams, streams)
            for _ in range(iterations):
                mix = mix - torch.logsumexp(mix, -1, keepdim=True)
                mix = mix - torch.logsumexp(mix, -2, keepdim=True)
            mix = mix.exp()
        branch = (pre[:, None] @ x).squeeze(1)
        pre_scale = self.loop_mhc.get("pre_scale", 1.0)
        if pre_scale != 1.0:
            branch = branch * pre_scale
        return norm(branch).view(batch, length, -1), mix.to(x.dtype), post

    branch, mix, post = route(residual, self.loop_attn, self.input_layernorm)
    if self.layer_type == "linear_attention":
        branch = self.linear_attn(branch, attention_mask=attention_mask, cache_params=None)
    else:
        branch = self.self_attn(
            branch, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings
        )[0]
    residual = mix @ residual + post[:, :, None] * branch.reshape(batch * length, 1, -1)
    branch, mix, post = route(residual, self.loop_ffn, self.post_attention_layernorm)
    branch = self.mlp(branch).reshape(batch * length, 1, -1)
    residual = mix @ residual + post[:, :, None] * branch
    result = residual.mean(1) if self.loop_exit else residual.flatten(1)
    return result.view(batch, length, -1)


@pytest.mark.parametrize("family", ["deepseek2", "deepseek2_lora", "deepseek3_yarn"])
@pytest.mark.parametrize(
    "prefill,decode",
    [
        ("triton", "triton"),
        ("fa3", "fa3"),
        ("fa3", "triton"),
        ("flashinfer", "flashinfer"),
        ("fa3", "flashinfer"),
        ("flashinfer", "fa3"),
    ],
)
def test_tiny_mla_model(family, prefill, decode, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(family, prefill, decode, "None", monkeypatch, tmp_path)


@pytest.mark.parametrize("family", ["deepseek3_moe", "glm4_moe_lite"])
@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("fused_shared", [False, True])
def test_tiny_mla_moe_model(family, implementation, fused_shared, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family,
        implementation,
        implementation,
        "None",
        monkeypatch,
        tmp_path,
        fused_shared_experts=fused_shared,
    )


@pytest.mark.parametrize("family", ["deepseek3_moe", "glm4_moe_lite"])
@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_tiny_mla_mtp_verification(family, implementation, layout, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(
        family, implementation, implementation, "None", monkeypatch, tmp_path, verify_layout=layout
    )


@pytest.mark.parametrize(
    "family,prefill,decode,kv_type",
    [
        ("deepseek3_2", "triton", "triton", "None"),
        ("deepseek3_2", "triton", "triton", "fp8kv_dsa"),
        ("deepseek3_2_flashmla", "flashmla_sparse", "triton", "None"),
        ("deepseek3_2_flashmla", "flashmla_sparse", "flashmla_sparse", "None"),
        ("deepseek3_2", "triton", "flashmla_sparse", "None"),
        ("deepseek3_2", "triton", "flashmla_sparse", "fp8kv_dsa"),
        ("deepseek3_2", "flashmla_sparse", "triton", "fp8kv_dsa"),
        ("deepseek3_2", "flashmla_sparse", "flashmla_sparse", "fp8kv_dsa"),
    ],
)
def test_tiny_sparse_mla_model(family, prefill, decode, kv_type, monkeypatch, tmp_path):
    test_tiny_model_prefill_decode_prefix_and_graph(family, prefill, decode, kv_type, monkeypatch, tmp_path)


class GroupQuantizedReferenceCache(DynamicCache):
    """Independent Torch quantization after RoPE, at the HF cache update boundary."""

    def __init__(self, bits, group_size):
        super().__init__()
        self.bits, self.group_size = bits, group_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        def quantize(values):
            shape = values.shape
            groups = values.float().reshape(*shape[:-1], -1, self.group_size)
            limit = 7 if self.bits == 4 else 127
            scale = (groups.abs().amax(-1, keepdim=True) / limit).to(values.dtype).float()
            integers = torch.where(scale == 0, 0, torch.trunc(groups / scale)).clamp(-limit, limit)
            return (integers * scale).reshape(shape).to(values.dtype)

        return super().update(quantize(key_states), quantize(value_states), layer_idx, cache_kwargs)


def absorbed_mla_reference(self, hidden_states, position_embeddings, attention_mask, **kwargs):
    """Independent Torch low-rank factorization, with the same BF16 rounding boundaries."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
    )

    batch, count, _ = hidden_states.shape
    query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    query = query.view(batch, count, self.num_heads, self.qk_head_dim).transpose(1, 2)
    nope, rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
    compressed = self.kv_a_proj_with_mqa(hidden_states)
    latent, key_rope = compressed.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
    latent = self.kv_a_layernorm(latent)
    rotate = apply_rotary_pos_emb_interleave if self.config.rope_interleave else apply_rotary_pos_emb
    rope, key_rope = rotate(rope, key_rope[:, None], *position_embeddings)
    projection = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
    query_latent = torch.einsum("bhtd,hdl->bhtl", nope, projection[:, : self.qk_nope_head_dim])
    scores = query_latent.float() @ latent[:, None].float().transpose(-1, -2)
    scores += rope.float() @ key_rope.float().transpose(-1, -2)
    scores *= self.scaling
    mask = torch.ones(count, count, dtype=torch.bool, device=hidden_states.device).triu(1)
    probabilities = scores.masked_fill(mask, -torch.inf).softmax(-1).to(latent.dtype)
    output = probabilities @ latent[:, None]
    values = torch.einsum("bhtl,hvl->bhtv", output, projection[:, self.qk_nope_head_dim :])
    return self.o_proj(values.transpose(1, 2).reshape(batch, count, -1)), None


class FP8ReferenceCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        def quantize(values):
            encoded = (values.float() / 0.0078125).clamp(-448, 448).to(torch.float8_e4m3fn)
            return (encoded.float() * 0.0078125).to(values.dtype)

        return super().update(quantize(key_states), quantize(value_states), layer_idx, cache_kwargs)


def prepare_model_batch(model, sequences, cached, prefill, *, input_only=False):
    from lightllm.common.basemodel.batch_objs import ModelInput

    lengths = [len(s) for s in sequences]
    new = [s[n:] for s, n in zip(sequences, cached)]
    tensor = lambda data, dtype=torch.int32: torch.tensor(data, dtype=dtype, device="cuda")
    inp = ModelInput(
        batch_size=2,
        total_token_num=sum(lengths),
        max_q_seq_len=max(map(len, new)),
        max_kv_seq_len=max(lengths),
        max_cache_len=max(cached),
        input_ids=tensor(sum(new, []), torch.int64),
        b_req_idx=tensor([0, 1]),
        b_seq_len=tensor(lengths),
        b_mtp_index=tensor([0, 0]),
        is_prefill=prefill,
        b_ready_cache_len=tensor(cached) if prefill else None,
        b_prefill_start_loc=tensor([0, len(new[0])]) if prefill else None,
        b_is_decode_req=tensor([False, False], torch.bool) if prefill else None,
        b_prefill_has_output_cpu=[True, True] if prefill else None,
        b_position_delta=None if prefill else tensor([0, 0]),
        b_shared_seq_len=None if prefill else tensor([0, 0]),
        b_shared_radix_node_id=None if prefill else tensor([0, 0]),
        multimodal_params=[{"images": [], "audios": []}, {"images": [], "audios": []}],
    )
    if input_only:
        return inp
    batch = model._create_inferstate(inp)
    model.batch_metadata(batch)
    batch.init_att_state()
    return batch
