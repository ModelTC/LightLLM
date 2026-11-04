from lightllm.core.compute.attention.dense.backend import DenseStaticParams
from lightllm.core.contracts.device import ExecutionLimits
from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
from lightllm.core.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.core.compute.attention.sparse_mla.spec import SparseMLAAttentionSpec
from lightllm.core.compute.attention.gdn.backend import GDNStaticParams
from lightllm.core.compute.attention.mla.backend import MLAStaticParams
from lightllm.core.compute.attention.sparse_mla.backend import SparseMLAStaticParams
from lightllm.core.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.core.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
from lightllm.core.compute.attention.kda.backend import KDAStaticParams
from lightllm.core.compute.attention.pooled_nope.backend import PooledNoPEStaticParams
from lightllm.core.build import BuildConfig, prepare
from lightllm.core.registry import default_providers

RECURRENT_SPECS = (GatedDeltaNetSpec, KimiDeltaAttentionSpec)


def prepare_runner(model):
    # Explicit opt-in keeps inherited recipes with different semantics on their
    # existing path until their adapters declare those differences.
    adapter = getattr(model, "model_adapter", None)
    if adapter is None:
        return None
    if model.args.llm_kv_type not in ("None", "int4kv", "int8kv", "fp8kv_sph", "fp8kv_spt", "fp8kv_dsa"):
        return None
    phases = adapter.attention_phases
    if len(set(phases)) != len(phases) or any(phase not in ("prefill", "decode") for phase in phases):
        raise ValueError("model attention phases must be unique prefill/decode stages")
    requested = {phase: getattr(model.args, f"llm_{phase}_att_backend")[0] for phase in phases}
    verify_tokens = model.mtp_manager.get_decode_tokens_per_request(model.is_mtp_draft_model)
    if verify_tokens > 1 and "decode" in requested:
        requested["verify"] = requested.pop("decode")
    if any(name not in ("auto", "triton", "fa3", "flashinfer", "flashmla_sparse") for name in requested.values()):
        return None
    specs = adapter.attention_specs()
    mla = any(isinstance(spec, MLAAttentionSpec) for spec in specs.values())
    sparse = any(isinstance(spec, SparseMLAAttentionSpec) for spec in specs.values())
    pooled = any(isinstance(spec, PooledNoPEAttentionSpec) for spec in specs.values())
    if sparse and model.args.llm_kv_type not in ("None", "fp8kv_dsa"):
        raise ValueError("sparse MLA requires plain or fp8kv_dsa storage")
    if not sparse and not pooled and model.args.llm_kv_type == "fp8kv_dsa":
        return None
    if pooled and model.args.llm_kv_type not in ("None", "fp8kv_dsa"):
        raise ValueError("pooled NoPE requires plain or group64/E8M0 packed KV")
    if (
        mla
        and not sparse
        and (
            model.args.llm_kv_type != "None"
            or any(not hasattr(weight, "cc_kv_b_proj_") for weight in model.trans_layers_weight)
        )
    ):
        return None
    from lightllm.core.device.cuda import CudaRuntime

    runtime = CudaRuntime()
    # Draft KV catch-up also consumes the target's expanded verification rows.
    capacity_tokens = max(verify_tokens, model.mtp_manager.get_decode_tokens_per_request(False))
    batch_capacity = max((model.max_req_num + 1) * capacity_tokens, model.graph_max_batch_size)
    if model.args.enable_tpsp_mix_mode:
        batch_capacity = (batch_capacity + model.tp_world_size_ - 1) // model.tp_world_size_ * model.tp_world_size_
    limits = ExecutionLimits(
        runtime.workspace_budget(model.mem_fraction),
        batch_capacity,
        model.get_req_manager_max_seq_len(),
        2 if model.args.enable_decode_microbatch_overlap or model.args.enable_prefill_microbatch_overlap else 1,
        max_verify_tokens=verify_tokens,
    )
    import torch

    from lightllm.core.formats.kv.config import history_formats
    from lightllm.core.formats.recurrent.conv_ssm import recurrent_format as make_recurrent_format

    calibration = None
    if model.args.llm_kv_type in ("fp8kv_sph", "fp8kv_spt"):
        from lightllm.core.formats.kv.calibration import calibration_table_identity

        calibration = calibration_table_identity(
            model.att_state_manager.read_full_att_calibration(model.args),
            "per_head" if model.args.llm_kv_type == "fp8kv_sph" else "per_tensor",
        )
    format, window_format = history_formats(
        specs,
        kv_type=model.args.llm_kv_type,
        model_dtype=model.data_type,
        page_size=model.args.page_size,
        shard=(model.tp_rank_, model.tp_world_size_),
        group_size=model.args.llm_kv_quant_group_size,
        calibration=calibration,
        capacity_tokens=capacity_tokens,
    )
    recurrent_format, recurrent_requested = None, None
    if any(isinstance(spec, RECURRENT_SPECS) for spec in specs.values()):
        from lightllm.utils.torch_dtype_utils import get_torch_dtype

        recurrent_format = make_recurrent_format(
            model.data_type,
            get_torch_dtype(model.args.linear_att_ssm_data_type),
            (model.tp_rank_, model.tp_world_size_),
            verify_tokens,
        )
        recurrent_requested = {}
        for phase in requested:
            names = getattr(model.args, f"llm_{'decode' if phase == 'verify' else phase}_att_backend")
            recurrent_requested[phase] = names[1] if len(names) > 1 else "auto"
    weights = {
        recipe.layer_num_: weight for recipe, weight in zip(model.layers_infer, model.trans_layers_weight, strict=True)
    }
    params = {
        i: GDNStaticParams.from_weights(spec, weights[i])
        if isinstance(spec, GatedDeltaNetSpec)
        else KDAStaticParams.from_weights(spec, weights[i])
        if isinstance(spec, KimiDeltaAttentionSpec)
        else PooledNoPEStaticParams(spec, runtime.allocate_tensor, weights[i].index_kpool_compress_ape.weight)
        if isinstance(spec, PooledNoPEAttentionSpec)
        else SparseMLAStaticParams(spec, runtime.allocate_tensor)
        if isinstance(spec, SparseMLAAttentionSpec)
        else MLAStaticParams(spec, runtime.allocate_tensor, weights[i].cc_kv_b_proj_.mm)
        if isinstance(spec, MLAAttentionSpec)
        else DenseStaticParams(
            spec, runtime.allocate_tensor, weights[i].attn_sinks.weight if spec.attention_sink else None
        )
        for i, spec in specs.items()
    }
    return prepare(
        model_adapter=adapter,
        providers=default_providers(),
        device=runtime,
        config=BuildConfig(
            layers=model.layers_infer,
            manager=model.att_state_manager,
            limits=limits,
            format=format,
            requested=requested,
            recurrent_format=recurrent_format,
            recurrent_requested=recurrent_requested,
            static_params=params,
            window_format=window_format,
        ),
    )
