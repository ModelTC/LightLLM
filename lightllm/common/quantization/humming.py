from dataclasses import replace
from functools import cached_property

import torch

from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune

from .registry import QUANTMETHODS
from .w8a8 import FP8w8a8MoEB32TritonQuantizationMethod

HUMMING_FP8_QUANT_METHOD = "fp8w8a8-moe-1x32-humming"
HUMMING_FP4_QUANT_METHOD = "mxfp4w4a8-moe-1x32-humming"
HUMMING_QUANT_METHODS = frozenset((HUMMING_FP8_QUANT_METHOD, HUMMING_FP4_QUANT_METHOD))

_GROUPED_CONTIGUOUS = '{"gemm_type":"grouped_contiguous","use_m_major_input_scale":true}'


def _humming_decode_configs():
    return (
        [{"rows_per_expert": 0}]
        + [{"rows_per_expert": rows} for rows in (8, 16, 32, 48, 64, 96, 128)]
        + [{"rows_per_expert": 64, "use_fp4_schedule": True}]
    )


def _humming_decode_key(gemm, expert_alignment):
    config = gemm.config
    return {
        "humming_version": gemm.version,
        "weight_dtype": str(config.b_dtype),
        "experts": config.num_experts,
        "n": config.shape_n,
        "k": config.shape_k,
        "expert_alignment": expert_alignment,
    }


def _humming_decode_tuning_inputs(gemm, inputs, scales, output, offsets, valid_rows, expert_alignment, locks):
    # Decode capture warmup can route dummy requests to only a few experts.
    # Benchmark balanced, aligned segments with scratch inputs and Stream-K locks.
    experts = gemm.config.num_experts
    rows_per_expert = max(1, (valid_rows + experts - 1) // experts)
    padded_rows = (rows_per_expert + expert_alignment - 1) // expert_alignment * expert_alignment
    rows = experts * padded_rows
    device = inputs.device
    benchmark_inputs = torch.zeros((rows, inputs.shape[1]), device=device, dtype=inputs.dtype)
    benchmark_scales = torch.ones((scales.shape[0], rows), device=device, dtype=scales.dtype)
    benchmark_output = torch.empty((rows, output.shape[1]), device=device, dtype=output.dtype)
    benchmark_offsets = torch.arange(experts + 1, device=device, dtype=offsets.dtype) * padded_rows
    return (
        gemm,
        benchmark_inputs,
        benchmark_scales,
        benchmark_output,
        benchmark_offsets,
        valid_rows,
        expert_alignment,
        torch.zeros_like(locks),
    ), {}


@autotune(
    kernel_name="humming_grouped_decode_gemm:v1",
    configs_gen_func=_humming_decode_configs,
    static_key_func=_humming_decode_key,
    run_key_func=lambda valid_rows: valid_rows,
    kernel_type=AutotuneKernelType.DECODE,
    rebuild_input_func=_humming_decode_tuning_inputs,
    warmup_all_exist_config=False,
)
def _humming_grouped_decode_gemm(
    gemm, inputs, scales, output, offsets, valid_rows, expert_alignment, locks, run_config=None
):
    tuning_config = None
    if run_config is not None and run_config["rows_per_expert"]:
        from humming.tune import get_heuristics_config

        # The FP4 policy also proposes tiles that can run with FP8 weights,
        # including the M=64 schedule that won the H200 cross-check.
        config = gemm.fp4_schedule_config if run_config.get("use_fp4_schedule") else gemm.config
        tuning_config = get_heuristics_config(
            config,
            shape_m=run_config["rows_per_expert"] * config.num_experts,
            gemm_type="grouped_contiguous",
            use_m_major_input_scale=True,
            device=inputs.device,
        )
    gemm._forward(inputs, scales, output, offsets, valid_rows, locks, tuning_config)


class HummingExpertGemm:
    def __init__(self, pack, weight_dtype, output_dtype, locks):
        if output_dtype != torch.bfloat16:
            raise ValueError("Humming V4.1 experts require BF16 output to preserve the UE8M0 exponent range")
        try:
            from humming import __version__
            from humming.forward import humming_forward
            from humming.schema import HummingInputSchema, HummingWeightSchema
            from humming.transform import prepare_layer_config, transform_humming_tensors
        except ImportError as exc:
            raise RuntimeError("Humming EP MoE requires humming-kernels==0.1.18") from exc

        experts, n, _ = pack.weight.shape
        k = pack.weight_scale.shape[-1] * 32
        config = prepare_layer_config(
            n,
            k,
            HummingWeightSchema(b_dtype=weight_dtype, bs_dtype="float8e8m0", weight_scale_group_size=32),
            HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=32, input_scale_dtype="float32"),
            num_experts=experts,
            torch_dtype=output_dtype,
            device=pack.weight.device,
        )
        # Preserve the checkpoint's full UE8M0 exponent range. Humming's fused
        # FP4 dequantization rebases scales and narrows that range.
        if config.use_fused_e8m0_scale:
            config = replace(config, use_fused_e8m0_scale=False, weight_scale_2_type="none")
        tensors = transform_humming_tensors(
            config,
            {
                "weight": pack.weight.view(torch.int32),
                "weight_scale": pack.weight_scale.view(torch.float8_e8m0fnu),
            },
        )
        # Reuse checkpoint storage after loading. All expert slices share it.
        for name, tensor in tensors.items():
            storage = getattr(pack, name).view(torch.uint8).flatten()
            storage.copy_(tensor.view(torch.uint8).flatten())
            tensors[name] = storage.view(tensor.dtype).view(tensor.shape)

        self.config = config
        self.version = __version__
        self.tensors = tensors
        self.locks = locks
        self.forward = humming_forward

    @cached_property
    def fp4_schedule_config(self):
        from humming.dtypes import float4e2m1

        return replace(self.config, b_dtype=float4e2m1, use_fused_e8m0_scale=False)

    def __call__(self, inputs, scales, output, offsets, valid_rows, *, expert_alignment=None):
        # DeepEP and the SwiGLU quantizer already allocate the padded M-major pitch.
        m_pad = (inputs.shape[0] + 3) // 4 * 4
        assert scales.stride() == (1, m_pad), "Humming requires TMA-aligned column-major activation scales"
        scales = scales.as_strided((scales.shape[1], m_pad), (m_pad, 1))
        if expert_alignment is None:
            self._forward(inputs, scales, output, offsets, valid_rows, self.locks)
        else:
            _humming_grouped_decode_gemm(
                self, inputs, scales, output, offsets, valid_rows, expert_alignment, self.locks
            )

    def _forward(self, inputs, scales, output, offsets, valid_rows, locks, tuning_config=None):
        self.forward(
            self.config,
            inputs,
            input_scale=scales,
            outputs=output,
            expert_layout=offsets,
            valid_shape_m=valid_rows,
            compute_config=_GROUPED_CONTIGUOUS,
            tuning_config=tuning_config,
            locks=locks,
            **self.tensors,
        )


@QUANTMETHODS.register(HUMMING_FP8_QUANT_METHOD, platform="cuda")
class HummingFP8MoEQuantizationMethod(FP8w8a8MoEB32TritonQuantizationMethod):
    humming_weight_dtype = "float8e4m3"

    @property
    def method_name(self):
        return HUMMING_FP8_QUANT_METHOD

    def finalize_moe_weight(self, moe_weight):
        device = moe_weight.w13.weight.device
        locks = torch.zeros(
            torch.cuda.get_device_properties(device).multi_processor_count * 8, dtype=torch.int32, device=device
        )
        for pack in (moe_weight.w13, moe_weight.w2):
            pack.humming_gemm = HummingExpertGemm(pack, self.humming_weight_dtype, moe_weight.data_type_, locks)


@QUANTMETHODS.register(HUMMING_FP4_QUANT_METHOD, platform="cuda")
class HummingFP4MoEQuantizationMethod(HummingFP8MoEQuantizationMethod):
    humming_weight_dtype = "float4e2m1"
    weight_dtype = torch.uint8
    weight_pack_factor = 2

    @property
    def method_name(self):
        return HUMMING_FP4_QUANT_METHOD

    def load_weight(self, weight, weight_pack):
        if weight.dtype not in (torch.int8, torch.uint8) or weight.shape != weight_pack.weight.shape:
            raise ValueError("Humming MXFP4 experts require two serialized E2M1 values per byte")
        weight_pack.weight.copy_(weight.view(torch.uint8))
        weight_pack.load_ok[0] = True
