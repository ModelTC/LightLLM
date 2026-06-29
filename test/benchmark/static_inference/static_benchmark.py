"""Static forward benchmark for LightLLM model parts.

The entry uses synthetic token ids and measures forward-only TPS for prefill,
chunked prefill, decode, and MTP decode cases.
"""

import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import PretrainedConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.models import get_model
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel
from lightllm.models.mistral_mtp.model import MistralMTPModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.utils.config_utils import get_dtype, get_vocab_size
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import set_env_start_args


DEFAULT_BATCH_SIZES = [2, 8, 16, 32, 64, 128]
MTP_MODES = {"vanilla_with_att", "eagle_with_att", "vanilla_no_att", "eagle_no_att"}
PREFILL_TABLE_HEADERS = [
    "ctx",
    "hit",
    "bs",
    "max_total_token_num",
    "uncached",
    "cached",
    "tokens",
    "ms",
    "qps",
    "tok/s",
    "logical_tok/s",
]
DECODE_TABLE_HEADERS = [
    "ctx",
    "bs",
    "accept",
    "max_total_token_num",
    "ms",
    "qps",
    "tok/s",
    "itl_ms",
]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    stage: str
    batch_size: int
    context_len: int
    output_len: int
    chunked_prefill_size: Optional[int] = None
    profiled_max_total_token_num: Optional[int] = None
    profiled_batch_divisor: Optional[int] = None
    cache_hit_rate: float = 0.0
    prefill_uncached_len: Optional[int] = None
    prefill_step_tokens_per_req: Optional[int] = None
    prefill_batch_size_by_batch_max_tokens: Optional[int] = None


@dataclass
class BenchmarkResult:
    case: str
    stage: str
    batch_size: int
    context_len: int
    output_len: int
    chunked_prefill_size: Optional[int]
    elapsed_ms: float
    measured_tokens: int
    qps: float
    tps: float
    profiled_max_total_token_num: Optional[int] = None
    profiled_batch_divisor: Optional[int] = None
    ttft_ms: Optional[float] = None
    inter_token_latency_ms: Optional[float] = None
    cache_hit_rate: float = 0.0
    prefill_uncached_len: Optional[int] = None
    prefill_cached_len: Optional[int] = None
    prefill_step_tokens_per_req: Optional[int] = None
    mtp_accept_rate: Optional[float] = None
    logical_tps: Optional[float] = None


class TokenSource:
    def __init__(self, args: SimpleNamespace):
        self.vocab_size = max(2, int(get_vocab_size(args.model_dir) or 0))
        self.rng = np.random.default_rng(args.seed)

    def batch(self, batch_size: int, need_len: int) -> np.ndarray:
        return self.rng.integers(low=0, high=self.vocab_size, size=(batch_size, need_len), dtype=np.int64)


def cpu_i32_full(shape, value) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.int32, device="cpu")


def cpu_i32_zeros(size: int) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.int32, device="cpu")


def empty_multimodal_params(batch_size: int) -> List[Dict]:
    return [{"images": [], "audios": []} for _ in range(batch_size)]


class StaticBenchmarkExecutor:
    def __init__(
        self,
        args: SimpleNamespace,
        model,
        draft_models: List,
        token_source: TokenSource,
    ):
        self.args = args
        self.model = model
        self.draft_models = draft_models
        self.token_source = token_source
        self.dp = int(args.dp or 1)

    def _case_iters(self, warmup: bool) -> int:
        return self.args.warmup_iters if warmup else self.args.bench_iters

    def run_case(self, case: BenchmarkCase, warmup: bool) -> BenchmarkResult:
        if case.stage == "prefill":
            return self._run_prefill_case(case, warmup)
        if case.stage == "decode":
            return self._run_decode_case(case, warmup)
        raise ValueError(f"unknown benchmark stage: {case.stage}")

    def _run_prefill_case(self, case: BenchmarkCase, warmup: bool) -> BenchmarkResult:
        """Measure full uncached prefill, chunked by production admission size."""
        uncached_len = int(case.prefill_uncached_len or case.context_len)
        cached_len = max(0, case.context_len - uncached_len)
        tokens = self.token_source.batch(case.batch_size, uncached_len)
        elapsed = 0.0
        measured_tokens = case.batch_size * uncached_len

        for _ in range(self._case_iters(warmup)):
            self._reset_model_cache()
            req_idx = self._alloc_req_indexes(case.batch_size)
            if cached_len > 0:
                self._materialize_cached_prefix(req_idx, cached_len)
            inputs = self._build_prefill_inputs(
                token_rows=tokens,
                req_idx=req_idx,
                prompt_len=uncached_len,
                chunk_size=case.chunked_prefill_size,
                initial_ready_cache_len=cached_len,
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = None
            for model_input in inputs:
                output = self._forward_prefill_input(model_input, allow_overlap=True)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            self._touch_output(output)

        self._reset_model_cache()
        return self._make_result(case, elapsed, measured_tokens, warmup)

    def _run_decode_case(self, case: BenchmarkCase, warmup: bool) -> BenchmarkResult:
        mtp_enabled = self._mtp_enabled()
        token_rows = self.token_source.batch(case.batch_size, case.context_len) if mtp_enabled else None
        measured_tokens = case.batch_size * case.output_len
        elapsed = 0.0
        ttft_elapsed = 0.0
        decode_step_count = 0
        iters = self._case_iters(warmup)

        for _ in range(iters):
            self._reset_model_cache()
            if mtp_enabled:
                torch.cuda.synchronize()
                ttft_start = time.perf_counter()
                req_idx, seq_len, next_ids = self._prefill_for_decode(case, token_rows, mtp_enabled)
                torch.cuda.synchronize()
                ttft_elapsed += time.perf_counter() - ttft_start
                step_elapsed, step_count = self._run_mtp_decode_steps(
                    case=case,
                    req_idx=req_idx,
                    seq_len=seq_len,
                    next_ids=next_ids,
                )
                elapsed += step_elapsed
                decode_step_count += step_count
            else:
                req_idx, seq_len, next_ids = self._materialize_context_for_decode(case)
                elapsed += self._run_plain_decode_steps(
                    case=case,
                    req_idx=req_idx,
                    seq_len=seq_len,
                    next_ids=next_ids,
                )
                decode_step_count += case.output_len

        self._reset_model_cache()
        inter_token_latency_ms = elapsed * 1000.0 / max(1, decode_step_count) if iters > 0 else None
        return self._make_result(
            case,
            elapsed,
            measured_tokens,
            warmup,
            ttft_elapsed_s=ttft_elapsed if mtp_enabled else None,
            inter_token_latency_ms=inter_token_latency_ms,
        )

    def _materialize_context_for_decode(self, case: BenchmarkCase):
        """Allocate historical KV slots so decode can be measured without prefill."""
        req_idx = self._alloc_req_indexes(case.batch_size)
        self._materialize_cached_prefix(req_idx, case.context_len)
        seq_len = cpu_i32_full((case.batch_size,), case.context_len)
        next_ids = torch.from_numpy(np.ascontiguousarray(self.token_source.batch(case.batch_size, 1).reshape(-1))).to(
            torch.int64
        )
        return req_idx, seq_len, next_ids

    def _prefill_for_decode(self, case: BenchmarkCase, token_rows: np.ndarray, mtp_enabled: bool):
        req_idx = self._alloc_req_indexes(case.batch_size)
        inputs = self._build_prefill_inputs(
            token_rows=token_rows,
            req_idx=req_idx,
            prompt_len=case.context_len,
            chunk_size=None,
        )
        output = None
        for model_input in inputs:
            output = self._forward_prefill_input(model_input, allow_overlap=not mtp_enabled)
        assert output is not None
        self._touch_output(output)

        next_ids = self._argmax_ids(output.logits)

        seq_len = cpu_i32_full((case.batch_size,), case.context_len)
        if mtp_enabled:
            next_ids = self._fill_mtp_prefill_kv(case, inputs[-1], output, next_ids)
        return req_idx, seq_len, next_ids

    def _fill_mtp_prefill_kv(
        self,
        case: BenchmarkCase,
        main_prefill_input: ModelInput,
        main_output: ModelOutput,
        first_next_ids: torch.Tensor,
    ):
        draft_input = main_prefill_input
        draft_output = main_output
        current_next_ids = first_next_ids.cuda(non_blocking=True)
        mtp_candidates = [current_next_ids.detach().cpu()]
        for draft_index in range(self._num_mtp_modules()):
            draft_input = prepare_mtp_prefill_inputs(
                model_input=draft_input,
                b_next_token_ids=current_next_ids,
                mtp_draft_input_hiddens=draft_output.mtp_main_output_hiddens,
            )
            draft_output = self.draft_models[draft_index].forward(draft_input)
            current_next_ids = self._argmax_ids(draft_output.logits).cuda(non_blocking=True)
            mtp_candidates.append(current_next_ids.detach().cpu())

        step_width = self._mtp_step_width()
        while len(mtp_candidates) < step_width:
            mtp_candidates.append(mtp_candidates[-1])
        next_ids = torch.stack(mtp_candidates[:step_width], dim=1)
        return next_ids

    def _run_plain_decode_steps(
        self,
        case: BenchmarkCase,
        req_idx: torch.Tensor,
        seq_len: torch.Tensor,
        next_ids: torch.Tensor,
    ) -> float:
        elapsed = 0.0
        for step in range(case.output_len):
            seq_len += 1
            model_input = self._make_decode_input(
                batch_size=case.batch_size,
                req_idx=req_idx,
                mtp_index=cpu_i32_zeros(case.batch_size),
                seq_len=seq_len,
                input_ids=next_ids.reshape(-1),
                max_kv_seq_len=int(seq_len.max().item()),
                mem_token_num=case.batch_size,
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = self._forward_decode_input(model_input, allow_overlap=True)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            self._touch_output(output)

            next_ids = self._argmax_ids(output.logits)
        return elapsed

    def _run_mtp_decode_steps(
        self,
        case: BenchmarkCase,
        req_idx: torch.Tensor,
        seq_len: torch.Tensor,
        next_ids: torch.Tensor,
    ) -> tuple:
        elapsed = 0.0
        step_count = 0
        generated_len = 0
        step_width = self._mtp_step_width()
        base_req_idx, b_mtp_index = self._build_mtp_decode_index_tensors(req_idx, step_width)
        current_candidates = next_ids

        while generated_len < case.output_len:
            accepted_width = self._sample_mtp_accept_width(step_width, case.output_len - generated_len)
            if current_candidates.ndim == 1:
                current_candidates = current_candidates[:, None].repeat(1, step_width)

            b_seq_len = self._build_mtp_seq_len(seq_len, step_width)
            model_input = self._make_decode_input(
                batch_size=case.batch_size * step_width,
                req_idx=base_req_idx,
                mtp_index=b_mtp_index,
                seq_len=b_seq_len,
                input_ids=current_candidates.reshape(-1),
                max_kv_seq_len=int(b_seq_len.max().item()),
                mem_token_num=case.batch_size * step_width,
            )

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = self.model.forward(model_input)
            candidate_rows, temporary_mem = self._run_mtp_draft_decode(
                model_input=model_input,
                model_output=output,
                real_batch_size=case.batch_size,
                step_width=step_width,
            )
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            self._touch_output(output)
            if temporary_mem is not None:
                self.model.req_manager.mem_manager.free(temporary_mem)

            self._free_rejected_mtp_mem(
                model_input=model_input,
                real_batch_size=case.batch_size,
                step_width=step_width,
                accepted_width=accepted_width,
            )
            current_candidates = (
                self._select_mtp_candidates(
                    candidate_rows=candidate_rows,
                    real_batch_size=case.batch_size,
                    step_width=step_width,
                    accepted_width=accepted_width,
                )
                .detach()
                .cpu()
            )
            seq_len += accepted_width
            generated_len += accepted_width
            step_count += 1

        return elapsed, step_count

    def _run_mtp_draft_decode(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        real_batch_size: int,
        step_width: int,
    ):
        draft_input = model_input
        draft_output = model_output
        draft_next_ids = self._argmax_ids(model_output.logits).cuda(non_blocking=True)
        generated = [draft_next_ids.detach()]

        temporary_mem = None
        if self.args.mtp_mode.startswith("eagle"):
            extra_mem_cpu = self.model.req_manager.mem_manager.alloc(real_batch_size * self.args.mtp_step)
            temporary_mem = extra_mem_cpu
            extra_mem = extra_mem_cpu.cuda(non_blocking=True)
        else:
            extra_mem = None

        for step in range(self.args.mtp_step):
            draft_input.input_ids = draft_next_ids
            draft_input.mtp_draft_input_hiddens = draft_output.mtp_main_output_hiddens
            draft_model = self.draft_models[step % self._num_mtp_modules()]
            draft_output = draft_model.forward(draft_input)
            draft_next_ids = self._argmax_ids(draft_output.logits).cuda(non_blocking=True)
            generated.append(draft_next_ids.detach())

            if self.args.mtp_mode.startswith("eagle") and step + 1 < self.args.mtp_step:
                draft_input.b_seq_len += 1
                draft_input.max_kv_seq_len += 1
                mem_i = extra_mem[step * real_batch_size : (step + 1) * real_batch_size]
                draft_input.mem_indexes = torch.cat(
                    [
                        draft_input.mem_indexes.view(-1, step_width)[:, 1:],
                        mem_i.view(-1, 1),
                    ],
                    dim=1,
                ).reshape(-1)

        return torch.stack(generated[:step_width], dim=1), temporary_mem

    def _sample_mtp_accept_width(self, step_width: int, remaining_tokens: int) -> int:
        """Sample accepted MTP width outside the timed decode section."""
        accept_rate = float(self.args.mtp_accept_rate)
        accepted_width = 1
        for _ in range(step_width - 1):
            if self.token_source.rng.random() >= accept_rate:
                break
            accepted_width += 1
        return max(1, min(accepted_width, remaining_tokens))

    def _select_mtp_candidates(
        self,
        candidate_rows: torch.Tensor,
        real_batch_size: int,
        step_width: int,
        accepted_width: int,
    ) -> torch.Tensor:
        row_ids = torch.arange(real_batch_size, device=candidate_rows.device) * step_width + accepted_width - 1
        return candidate_rows.index_select(0, row_ids)

    def _free_rejected_mtp_mem(
        self,
        model_input: ModelInput,
        real_batch_size: int,
        step_width: int,
        accepted_width: int,
    ):
        if accepted_width >= step_width:
            return
        rejected_mem = (
            model_input.mem_indexes_cpu.view(real_batch_size, step_width)[:, accepted_width:].contiguous().reshape(-1)
        )
        if rejected_mem.numel() > 0:
            self.model.req_manager.mem_manager.free(rejected_mem)

    def _build_prefill_inputs(
        self,
        token_rows: np.ndarray,
        req_idx: torch.Tensor,
        prompt_len: int,
        chunk_size: Optional[int],
        initial_ready_cache_len: int = 0,
    ) -> List[ModelInput]:
        if not chunk_size or chunk_size <= 0 or chunk_size >= prompt_len:
            return [
                self._make_prefill_input(
                    token_rows[:, :prompt_len],
                    req_idx,
                    ready_cache_len=initial_ready_cache_len,
                )
            ]

        inputs = []
        for start in range(0, prompt_len, chunk_size):
            end = min(prompt_len, start + chunk_size)
            inputs.append(
                self._make_prefill_input(
                    token_rows[:, start:end],
                    req_idx,
                    ready_cache_len=initial_ready_cache_len + start,
                )
            )
        return inputs

    def _materialize_cached_prefix(self, req_idx: torch.Tensor, cached_len: int):
        """Allocate dummy prefix KV so cache-hit cases consume real capacity."""
        if cached_len <= 0:
            return
        batch_size = int(req_idx.shape[0])
        need_tokens = batch_size * cached_len
        mem_indexes = self.model.req_manager.mem_manager.alloc(need_tokens)
        if mem_indexes is None:
            raise RuntimeError(f"failed to allocate cached prefix: bs={batch_size} cached_len={cached_len}")
        req_idx_gpu = req_idx.cuda(non_blocking=True)
        mem_indexes_gpu = mem_indexes.reshape(batch_size, cached_len).cuda(non_blocking=True)
        self.model.req_manager.req_to_token_indexs[req_idx_gpu, :cached_len] = mem_indexes_gpu

    def _make_prefill_input(self, token_chunk: np.ndarray, req_idx: torch.Tensor, ready_cache_len: int) -> ModelInput:
        batch_size, q_len = token_chunk.shape
        seq_len_value = ready_cache_len + q_len
        b_seq_len = cpu_i32_full((batch_size,), seq_len_value)
        b_ready_cache_len = cpu_i32_full((batch_size,), ready_cache_len)
        b_q_seq_len = b_seq_len - b_ready_cache_len
        b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len
        input_ids = torch.from_numpy(np.ascontiguousarray(token_chunk.reshape(-1))).to(torch.int64)
        mem_indexes = self.model.req_manager.mem_manager.alloc(input_ids.shape[0])
        return ModelInput(
            batch_size=batch_size,
            total_token_num=int(b_seq_len.sum().item()),
            max_q_seq_len=q_len,
            max_kv_seq_len=seq_len_value,
            max_cache_len=ready_cache_len,
            prefix_total_token_num=ready_cache_len * batch_size,
            input_ids=input_ids,
            b_req_idx=req_idx,
            b_mtp_index=cpu_i32_zeros(batch_size),
            b_seq_len=b_seq_len,
            mem_indexes_cpu=mem_indexes,
            is_prefill=True,
            b_ready_cache_len=b_ready_cache_len,
            b_prefill_start_loc=b_prefill_start_loc,
            b_prefill_has_output_cpu=[False] * batch_size,
            multimodal_params=empty_multimodal_params(batch_size),
        )

    def _make_decode_input(
        self,
        batch_size: int,
        req_idx: torch.Tensor,
        mtp_index: torch.Tensor,
        seq_len: torch.Tensor,
        input_ids: torch.Tensor,
        max_kv_seq_len: int,
        mem_token_num: int,
    ) -> ModelInput:
        mem_indexes = self.model.req_manager.mem_manager.alloc(mem_token_num)
        return ModelInput(
            batch_size=batch_size,
            total_token_num=int(seq_len.sum().item()),
            max_q_seq_len=1,
            max_kv_seq_len=max_kv_seq_len,
            input_ids=input_ids.to(torch.int64).cpu(),
            b_req_idx=req_idx,
            b_mtp_index=mtp_index,
            b_seq_len=seq_len,
            mem_indexes_cpu=mem_indexes,
            is_prefill=False,
            multimodal_params=empty_multimodal_params(batch_size),
        )

    def _forward_prefill_input(self, model_input: ModelInput, allow_overlap: bool) -> ModelOutput:
        if allow_overlap and self.args.enable_prefill_microbatch_overlap and model_input.batch_size > 1:
            micro_input0, micro_input1 = self._split_prefill_input(model_input)
            output0, output1 = self.model.microbatch_overlap_prefill(micro_input0, micro_input1)
            return self._merge_model_outputs(output0, output1)
        return self.model.forward(model_input)

    def _forward_decode_input(self, model_input: ModelInput, allow_overlap: bool) -> ModelOutput:
        if allow_overlap and self.args.enable_decode_microbatch_overlap and model_input.batch_size > 1:
            micro_input0, micro_input1 = self._split_decode_input(model_input)
            output0, output1 = self.model.microbatch_overlap_decode(micro_input0, micro_input1)
            return self._merge_model_outputs(output0, output1)
        return self.model.forward(model_input)

    def _split_prefill_input(self, model_input: ModelInput):
        split_batch = model_input.batch_size // 2
        q_lens = model_input.b_seq_len - model_input.b_ready_cache_len
        split_tokens = int(q_lens[:split_batch].sum().item())
        return (
            self._slice_prefill_input(model_input, 0, split_batch, 0, split_tokens),
            self._slice_prefill_input(
                model_input,
                split_batch,
                model_input.batch_size,
                split_tokens,
                int(q_lens.sum().item()),
            ),
        )

    def _slice_prefill_input(
        self,
        model_input: ModelInput,
        batch_start: int,
        batch_end: int,
        token_start: int,
        token_end: int,
    ) -> ModelInput:
        b_seq_len = model_input.b_seq_len[batch_start:batch_end].clone()
        b_ready_cache_len = model_input.b_ready_cache_len[batch_start:batch_end].clone()
        b_q_seq_len = b_seq_len - b_ready_cache_len
        b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len
        has_output = model_input.b_prefill_has_output_cpu
        return ModelInput(
            batch_size=batch_end - batch_start,
            total_token_num=int(b_seq_len.sum().item()),
            max_q_seq_len=int(b_q_seq_len.max().item()),
            max_kv_seq_len=int(b_seq_len.max().item()),
            max_cache_len=int(b_ready_cache_len.max().item()),
            prefix_total_token_num=int(b_ready_cache_len.sum().item()),
            input_ids=model_input.input_ids[token_start:token_end].contiguous(),
            b_req_idx=model_input.b_req_idx[batch_start:batch_end].clone(),
            b_mtp_index=model_input.b_mtp_index[batch_start:batch_end].clone(),
            b_seq_len=b_seq_len,
            mem_indexes_cpu=model_input.mem_indexes_cpu[token_start:token_end].contiguous(),
            is_prefill=True,
            b_ready_cache_len=b_ready_cache_len,
            b_prefill_start_loc=b_prefill_start_loc,
            b_prefill_has_output_cpu=(has_output[batch_start:batch_end] if has_output is not None else None),
            multimodal_params=model_input.multimodal_params[batch_start:batch_end],
        )

    def _split_decode_input(self, model_input: ModelInput):
        split_batch = model_input.batch_size // 2
        return (
            self._slice_decode_input(model_input, 0, split_batch),
            self._slice_decode_input(model_input, split_batch, model_input.batch_size),
        )

    def _slice_decode_input(self, model_input: ModelInput, batch_start: int, batch_end: int) -> ModelInput:
        b_seq_len = model_input.b_seq_len[batch_start:batch_end].clone()
        input_ids = model_input.input_ids
        if input_ids is not None:
            input_ids = input_ids[batch_start:batch_end].contiguous()
        return ModelInput(
            batch_size=batch_end - batch_start,
            total_token_num=int(b_seq_len.sum().item()),
            max_q_seq_len=model_input.max_q_seq_len,
            max_kv_seq_len=int(b_seq_len.max().item()),
            input_ids=input_ids,
            b_req_idx=model_input.b_req_idx[batch_start:batch_end].clone(),
            b_mtp_index=model_input.b_mtp_index[batch_start:batch_end].clone(),
            b_seq_len=b_seq_len,
            mem_indexes_cpu=model_input.mem_indexes_cpu[batch_start:batch_end].contiguous(),
            is_prefill=False,
            multimodal_params=model_input.multimodal_params[batch_start:batch_end],
        )

    def _merge_model_outputs(self, output0: ModelOutput, output1: ModelOutput) -> ModelOutput:
        mtp_hiddens = None
        if output0.mtp_main_output_hiddens is not None and output1.mtp_main_output_hiddens is not None:
            mtp_hiddens = torch.cat(
                (output0.mtp_main_output_hiddens, output1.mtp_main_output_hiddens),
                dim=0,
            )
        return ModelOutput(
            logits=torch.cat((output0.logits, output1.logits), dim=0),
            prefill_mem_indexes_ready_event=output0.prefill_mem_indexes_ready_event,
            mtp_main_output_hiddens=mtp_hiddens,
        )

    def _build_mtp_decode_index_tensors(self, req_idx: torch.Tensor, step_width: int):
        batch_size = int(req_idx.shape[0])
        return (
            req_idx.repeat_interleave(step_width).to(torch.int32).cpu(),
            torch.arange(step_width, dtype=torch.int32).repeat(batch_size),
        )

    def _build_mtp_seq_len(self, base_seq_len: torch.Tensor, step_width: int) -> torch.Tensor:
        offsets = torch.arange(1, step_width + 1, dtype=torch.int32)
        return (base_seq_len[:, None].to(torch.int32) + offsets[None, :]).reshape(-1)

    def _alloc_req_indexes(self, batch_size: int) -> torch.Tensor:
        req_indexes = [self.model.req_manager.alloc() for _ in range(batch_size)]
        if any(index is None for index in req_indexes):
            raise RuntimeError(f"failed to allocate {batch_size} request indexes")
        return torch.tensor(req_indexes, dtype=torch.int32, device="cpu")

    def _reset_model_cache(self):
        self.model.mem_manager.free_all()
        self.model.req_manager.free_all()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _argmax_ids(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1).detach().cpu().to(torch.int64)

    def _touch_output(self, output: Optional[ModelOutput]):
        if output is not None and output.logits is not None:
            _ = output.logits.shape

    def _make_result(
        self,
        case: BenchmarkCase,
        elapsed_s: float,
        measured_tokens: int,
        warmup: bool,
        ttft_elapsed_s: Optional[float] = None,
        inter_token_latency_ms: Optional[float] = None,
    ) -> BenchmarkResult:
        """Convert raw timings into reported TPS and latency metrics."""
        iters = self._case_iters(warmup)
        scaled_tokens = measured_tokens * self.dp * iters
        qps = case.batch_size * self.dp * iters / elapsed_s if elapsed_s > 0 else 0.0
        tps = scaled_tokens / elapsed_s if elapsed_s > 0 else 0.0
        ttft_ms = ttft_elapsed_s * 1000.0 / max(1, iters) if ttft_elapsed_s is not None else None
        logical_tps = None
        prefill_uncached_len = case.prefill_uncached_len
        prefill_cached_len = None
        if case.stage == "prefill":
            uncached_len = int(case.prefill_uncached_len or case.context_len)
            prefill_uncached_len = uncached_len
            prefill_cached_len = max(0, case.context_len - uncached_len)
            token_count = case.batch_size * case.context_len * self.dp * iters
            logical_tps = token_count / elapsed_s if elapsed_s > 0 else 0.0
        return BenchmarkResult(
            case=case.name,
            stage=case.stage,
            batch_size=case.batch_size,
            context_len=case.context_len,
            output_len=case.output_len,
            chunked_prefill_size=case.chunked_prefill_size,
            elapsed_ms=elapsed_s * 1000.0,
            measured_tokens=scaled_tokens,
            qps=qps,
            tps=tps,
            profiled_max_total_token_num=case.profiled_max_total_token_num,
            profiled_batch_divisor=case.profiled_batch_divisor,
            ttft_ms=ttft_ms,
            inter_token_latency_ms=inter_token_latency_ms,
            cache_hit_rate=case.cache_hit_rate,
            prefill_uncached_len=prefill_uncached_len,
            prefill_cached_len=prefill_cached_len,
            prefill_step_tokens_per_req=case.prefill_step_tokens_per_req,
            mtp_accept_rate=(
                float(self.args.mtp_accept_rate) if case.stage == "decode" and self._mtp_enabled() else None
            ),
            logical_tps=logical_tps,
        )

    def _mtp_enabled(self) -> bool:
        return self.args.mtp_mode in MTP_MODES and self.args.mtp_step > 0

    def _mtp_step_width(self) -> int:
        return int(self.args.mtp_step) + 1

    def _num_mtp_modules(self) -> int:
        if not self._mtp_enabled():
            return 0
        if self.args.mtp_mode.startswith("eagle"):
            return 1
        return int(self.args.mtp_step)


def parse_typed_list(value: Optional[str], fallback: Sequence, cast) -> List:
    if value is None or value == "":
        return list(fallback)
    if isinstance(value, cast):
        return [value]
    normalized = str(value).replace(",", " ")
    return [cast(item) for item in normalized.split() if item.strip()]


def parse_int_list(value: Optional[str], fallback: Sequence[int]) -> List[int]:
    return parse_typed_list(value, fallback, int)


def parse_float_list(value: Optional[str], fallback: Sequence[float]) -> List[float]:
    return parse_typed_list(value, fallback, float)


def parse_chunk_sizes(value: Optional[str], fallback: Optional[int]) -> List[Optional[int]]:
    if value is None:
        return [fallback] if fallback else [None]
    chunks: List[Optional[int]] = []
    for item in str(value).replace(",", " ").split():
        item = item.strip().lower()
        if item in {"none", "full", "0", "-1"}:
            chunks.append(None)
        else:
            chunks.append(int(item))
    return chunks or [None]


def prefill_uncached_len(context_len: int, cache_hit_rate: float) -> int:
    """Return the uncached suffix length for a prompt-cache hit ratio."""
    if cache_hit_rate < 0.0 or cache_hit_rate >= 1.0:
        raise ValueError(f"cache hit rate must satisfy 0 <= hit < 1, got {cache_hit_rate}")
    uncached = int(math.ceil(context_len * (1.0 - cache_hit_rate)))
    return max(1, min(context_len, uncached))


def prefill_step_tokens_per_req(uncached_len: int, chunked_prefill_size: Optional[int]) -> int:
    """Return tokens handled per request in one production prefill step."""
    if chunked_prefill_size and chunked_prefill_size > 0:
        return max(1, min(uncached_len, int(chunked_prefill_size)))
    return max(1, uncached_len)


def format_cache_hit_suffix(cache_hit_rate: float) -> str:
    return f"{cache_hit_rate:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def apply_max_batch_size(batch_size: int, max_batch_size: int) -> int:
    """Apply the benchmark-wide auto batch-size upper bound."""
    if max_batch_size > 0:
        batch_size = min(batch_size, int(max_batch_size))
    return max(1, batch_size)


def prefill_batch_size_from_batch_max_tokens(
    batch_max_tokens: int,
    step_tokens_per_req: int,
    max_batch_size: int,
) -> int:
    """Compute prefill BS from batch_max_tokens before KV-capacity capping."""
    batch_size = max(1, int(batch_max_tokens) // max(1, step_tokens_per_req))
    return apply_max_batch_size(batch_size, max_batch_size)


def build_prefill_cases(
    args: SimpleNamespace,
    input_lens: Sequence[int],
    chunk_sizes: Sequence[Optional[int]],
    cache_hit_rates: Sequence[float],
) -> List[BenchmarkCase]:
    """Build full-prefill cases using batch_max_tokens per chunk step."""
    if args.batch_max_tokens is None:
        raise ValueError("prefill benchmark requires --batch_max_tokens")
    cases: List[BenchmarkCase] = []
    for input_len in input_lens:
        for chunk_size in chunk_sizes:
            for cache_hit_rate in cache_hit_rates:
                uncached_len = prefill_uncached_len(input_len, cache_hit_rate)
                step_tokens = prefill_step_tokens_per_req(uncached_len, chunk_size)
                bs = prefill_batch_size_from_batch_max_tokens(
                    args.batch_max_tokens,
                    step_tokens,
                    args.max_batch_size,
                )
                chunk_name = chunk_size if chunk_size else "none"
                hit_name = format_cache_hit_suffix(cache_hit_rate)
                cases.append(
                    BenchmarkCase(
                        name=(
                            f"prefill_bs{bs}_in{input_len}_hit{hit_name}"
                            f"_uncached{uncached_len}_chunk{chunk_name}"
                            f"_btok{args.batch_max_tokens}"
                        ),
                        stage="prefill",
                        batch_size=bs,
                        context_len=input_len,
                        output_len=0,
                        chunked_prefill_size=chunk_size,
                        cache_hit_rate=cache_hit_rate,
                        prefill_uncached_len=uncached_len,
                        prefill_step_tokens_per_req=step_tokens,
                        prefill_batch_size_by_batch_max_tokens=bs,
                    )
                )
    return cases


def build_decode_cases(
    args: SimpleNamespace,
    batch_sizes: Sequence[int],
    context_lens: Sequence[int],
    output_lens: Sequence[int],
) -> List[BenchmarkCase]:
    """Build decode cases; profile mode resolves BS after model load."""
    decode_batch_sizes = [1] if args.decode_batch_size_mode == "profile" else batch_sizes
    cases: List[BenchmarkCase] = []
    for bs in decode_batch_sizes:
        for context_len in context_lens:
            for output_len in output_lens:
                profile_suffix = "_profilebs" if args.decode_batch_size_mode == "profile" else ""
                cases.append(
                    BenchmarkCase(
                        name=(f"decode_bs{bs}_ctx{context_len}_out{output_len}" f"{profile_suffix}"),
                        stage="decode",
                        batch_size=bs,
                        context_len=context_len,
                        output_len=output_len,
                    )
                )
    return cases


def build_cases(args: SimpleNamespace) -> List[BenchmarkCase]:
    """Expand CLI list options into concrete prefill/decode benchmark cases."""
    batch_sizes = parse_int_list(args.batch_sizes, [args.batch_size] if args.batch_size else DEFAULT_BATCH_SIZES)
    input_lens = parse_int_list(args.input_lens, [args.input_len])
    context_lens = parse_int_list(args.context_lens, input_lens)
    output_lens = parse_int_list(args.output_lens, [args.output_len])
    chunk_sizes = parse_chunk_sizes(args.chunked_prefill_sizes, args.chunked_prefill_size)
    cache_hit_rates = parse_float_list(args.prefill_cache_hit_rates, [0.0])

    cases: List[BenchmarkCase] = []
    if args.benchmark in {"all", "prefill"}:
        cases.extend(build_prefill_cases(args, input_lens, chunk_sizes, cache_hit_rates))
    if args.benchmark in {"all", "decode"}:
        cases.extend(build_decode_cases(args, batch_sizes, context_lens, output_lens))
    return cases


def decode_profile_batch_divisor(args: SimpleNamespace, case: BenchmarkCase) -> int:
    """Reserve KV capacity for context, generated tokens, and MTP expansion."""
    mtp_width = int(args.mtp_step) + 1 if args.mtp_mode in MTP_MODES else 1
    return max(1, case.context_len + case.output_len + mtp_width + 8)


def resolve_profile_decode_cases(
    args: SimpleNamespace,
    cases: Sequence[BenchmarkCase],
    profiled_max_total_token_num: int,
) -> List[BenchmarkCase]:
    """Replace decode profile placeholders with capacity-derived max BS."""
    if args.decode_batch_size_mode != "profile":
        return list(cases)

    resolved: List[BenchmarkCase] = []
    for case in cases:
        if case.stage != "decode":
            resolved.append(case)
            continue

        divisor = decode_profile_batch_divisor(args, case)
        batch_size = max(1, int(profiled_max_total_token_num) // divisor)
        batch_size = apply_max_batch_size(batch_size, args.max_batch_size)

        resolved.append(
            replace(
                case,
                name=(
                    f"decode_bs{batch_size}_ctx{case.context_len}"
                    f"_out{case.output_len}_profile{profiled_max_total_token_num}"
                ),
                batch_size=batch_size,
                profiled_max_total_token_num=int(profiled_max_total_token_num),
                profiled_batch_divisor=divisor,
            )
        )

    return resolved


def resolve_batch_max_prefill_cases(
    args: SimpleNamespace,
    cases: Sequence[BenchmarkCase],
    profiled_max_total_token_num: int,
) -> List[BenchmarkCase]:
    """Cap prefill BS by profiled KV capacity after the model is loaded."""
    resolved: List[BenchmarkCase] = []
    capacity_tokens = int(profiled_max_total_token_num)
    for case in cases:
        if case.stage != "prefill":
            resolved.append(case)
            continue

        if case.context_len <= 0:
            raise ValueError(f"invalid prefill context_len={case.context_len}")
        bs_by_capacity = capacity_tokens // case.context_len
        if bs_by_capacity <= 0:
            raise ValueError(
                "single prefill request does not fit profiled token capacity: "
                f"context_len={case.context_len} capacity={profiled_max_total_token_num}"
            )

        bs_by_batch = int(case.prefill_batch_size_by_batch_max_tokens or case.batch_size)
        batch_size = min(bs_by_batch, bs_by_capacity)
        batch_size = apply_max_batch_size(batch_size, args.max_batch_size)

        chunk_name = case.chunked_prefill_size if case.chunked_prefill_size else "none"
        hit_name = format_cache_hit_suffix(case.cache_hit_rate)
        resolved.append(
            replace(
                case,
                name=(
                    f"prefill_bs{batch_size}_in{case.context_len}_hit{hit_name}"
                    f"_uncached{case.prefill_uncached_len}_chunk{chunk_name}"
                    f"_btok{args.batch_max_tokens}"
                    f"_cap{profiled_max_total_token_num}"
                ),
                batch_size=batch_size,
                profiled_max_total_token_num=int(profiled_max_total_token_num),
                profiled_batch_divisor=case.context_len,
            )
        )

    return resolved


def normalize_args(args: argparse.Namespace, cases: Sequence[BenchmarkCase]) -> SimpleNamespace:
    """Fill LightLLM startup args needed before model construction."""
    if args.data_type is None:
        args.data_type = get_dtype(args.model_dir)

    if args.quant_type is None:
        args.quant_type = "none"

    if not 0.0 <= float(args.mtp_accept_rate) <= 1.0:
        raise ValueError(f"--mtp_accept_rate must be in [0, 1], got {args.mtp_accept_rate}")

    max_batch = max(case.batch_size for case in cases)
    max_context = max(case.context_len for case in cases)
    max_output = max(case.output_len for case in cases)
    mtp_width = (args.mtp_step + 1) if args.mtp_mode in MTP_MODES else 1
    max_runtime_len = max_context + max_output + mtp_width + 2

    if args.max_req_total_len is None:
        args.max_req_total_len = max_runtime_len
    else:
        args.max_req_total_len = max(args.max_req_total_len, max_runtime_len)

    if args.graph_max_len_in_batch == 0:
        args.graph_max_len_in_batch = args.max_req_total_len

    max_prefill_chunk = (
        max(
            min(case.context_len, case.chunked_prefill_size or case.context_len)
            for case in cases
            if case.stage == "prefill"
        )
        if any(case.stage == "prefill" for case in cases)
        else max_context
    )
    if args.batch_max_tokens is None:
        args.batch_max_tokens = max(max_batch * max_prefill_chunk, max_batch * mtp_width, 1)

    decode_batch_size_needs_profile = (
        args.benchmark in {"all", "decode"}
        and args.decode_batch_size_mode == "profile"
        and args.max_total_token_num is None
    )
    prefill_batch_size_needs_profile = args.benchmark in {"all", "prefill"} and args.max_total_token_num is None
    needs_profiled_batch_size = decode_batch_size_needs_profile or prefill_batch_size_needs_profile

    if args.max_total_token_num is None and not needs_profiled_batch_size:
        args.max_total_token_num = max_batch * (args.max_req_total_len + mtp_width + 8)
    if args.max_total_token_num is not None:
        args.max_total_token_num = max(args.max_total_token_num, args.batch_max_tokens + 1, args.max_req_total_len)

    if decode_batch_size_needs_profile and args.max_batch_size > 0:
        args.running_max_req_size = max(args.running_max_req_size, int(args.max_batch_size))
        # Profile decode BS is resolved after model load. Use the cap as the
        # pre-load upper bound so request slots and optional decode graphs agree.
        if not args.disable_cudagraph:
            args.graph_max_batch_size = max(args.graph_max_batch_size, int(args.max_batch_size))
    if prefill_batch_size_needs_profile:
        args.running_max_req_size = max(args.running_max_req_size, max_batch)

    if args.graph_max_batch_size < max_batch:
        args.graph_max_batch_size = max_batch

    if args.nccl_port is None:
        args.nccl_port = 28765

    if args.mtp_mode in MTP_MODES:
        if args.mtp_step <= 0:
            raise ValueError("--mtp_mode requires --mtp_step > 0")
        if not args.mtp_draft_model_dir:
            raise ValueError("--mtp_mode requires --mtp_draft_model_dir")
        args.mtp_draft_model_dir = normalize_mtp_draft_dirs(args.mtp_mode, args.mtp_step, args.mtp_draft_model_dir)
    else:
        args.mtp_mode = None
        args.mtp_step = 0
        args.mtp_draft_model_dir = None

    return SimpleNamespace(**vars(args))


def normalize_mtp_draft_dirs(mtp_mode: str, mtp_step: int, draft_dirs: Sequence[str]) -> List[str]:
    expected = 1 if mtp_mode.startswith("eagle") else mtp_step
    if isinstance(draft_dirs, str):
        draft_dirs = [draft_dirs]
    draft_dirs = list(draft_dirs)
    if len(draft_dirs) == 1 and expected > 1:
        return draft_dirs * expected
    if len(draft_dirs) != expected:
        raise ValueError(f"{mtp_mode} expects {expected} draft model dir(s), got {len(draft_dirs)}")
    return draft_dirs


def build_model_kvargs(args: SimpleNamespace, rank_id: int) -> Dict:
    return {
        "args": args,
        "nccl_host": args.nccl_host,
        "nccl_port": args.nccl_port,
        "rank_id": rank_id,
        "world_size": args.tp,
        "dp_size": args.dp,
        "weight_dir": args.model_dir,
        "data_type": args.data_type,
        "quant_type": args.quant_type,
        "quant_cfg": args.quant_cfg,
        "expert_dtype": args.expert_dtype,
        "load_way": "HF",
        "max_total_token_num": args.max_total_token_num,
        "graph_max_len_in_batch": args.graph_max_len_in_batch,
        "graph_max_batch_size": args.graph_max_batch_size,
        "mem_fraction": args.mem_fraction,
        "max_req_num": max(args.running_max_req_size, args.graph_max_batch_size),
        "batch_max_tokens": args.batch_max_tokens,
        "run_mode": "normal",
        "max_seq_length": args.max_req_total_len,
        "disable_cudagraph": args.disable_cudagraph,
        "llm_prefill_att_backend": args.llm_prefill_att_backend,
        "llm_decode_att_backend": args.llm_decode_att_backend,
        "vit_att_backend": args.vit_att_backend,
        "llm_kv_type": args.llm_kv_type,
        "llm_kv_quant_group_size": args.llm_kv_quant_group_size,
    }


def init_mtp_draft_models(args: SimpleNamespace, main_kvargs: Dict, main_model) -> List:
    if args.mtp_mode not in MTP_MODES:
        return []

    os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
    draft_models = []
    for draft_dir in args.mtp_draft_model_dir:
        mtp_cfg, _ = PretrainedConfig.get_config_dict(draft_dir)
        model_type = mtp_cfg.get("model_type", "")
        mtp_kvargs = {
            "weight_dir": draft_dir,
            "max_total_token_num": main_model.mem_manager.size,
            "load_way": main_kvargs["load_way"],
            "max_req_num": main_kvargs["max_req_num"],
            "max_seq_length": main_kvargs["max_seq_length"],
            "is_token_healing": False,
            "return_all_prompt_logics": False,
            "disable_chunked_prefill": args.disable_chunked_prefill,
            "data_type": main_kvargs["data_type"],
            "graph_max_batch_size": main_kvargs["graph_max_batch_size"],
            "graph_max_len_in_batch": main_kvargs["graph_max_len_in_batch"],
            "disable_cudagraph": main_kvargs["disable_cudagraph"],
            "mem_fraction": main_kvargs["mem_fraction"],
            "batch_max_tokens": main_kvargs["batch_max_tokens"],
            "quant_type": main_kvargs["quant_type"],
            "quant_cfg": main_kvargs["quant_cfg"],
            "expert_dtype": main_kvargs["expert_dtype"],
            "run_mode": "normal",
            "main_model": main_model,
            "mtp_previous_draft_models": draft_models.copy(),
        }
        if model_type == "deepseek_v3":
            assert args.mtp_mode in {
                "vanilla_with_att",
                "eagle_with_att",
            }, f"{model_type} MTP requires *_with_att mode"
            draft_models.append(Deepseek3MTPModel(mtp_kvargs))
        elif model_type == "qwen3_moe":
            assert args.mtp_mode in {
                "vanilla_no_att",
                "eagle_no_att",
            }, f"{model_type} MTP requires *_no_att mode"
            draft_models.append(Qwen3MOEMTPModel(mtp_kvargs))
        elif model_type == "mistral":
            assert args.mtp_mode in {
                "vanilla_no_att",
                "eagle_no_att",
            }, f"{model_type} MTP requires *_no_att mode"
            draft_models.append(MistralMTPModel(mtp_kvargs))
        elif model_type == "glm4_moe_lite":
            assert args.mtp_mode in {
                "vanilla_with_att",
                "eagle_with_att",
            }, f"{model_type} MTP requires *_with_att mode"
            draft_models.append(Glm4MoeLiteMTPModel(mtp_kvargs))
        else:
            raise ValueError(f"unsupported MTP draft model_type={model_type} from {draft_dir}")
    return draft_models


def run_worker(args_dict: Dict, case_dicts: List[Dict], rank_id: int, ans_queue):
    try:
        args = SimpleNamespace(**args_dict)
        cases = [BenchmarkCase(**case) for case in case_dicts]
        set_env_start_args(args)

        from lightllm.distributed import dist_group_manager
        import torch.distributed as dist

        model_kvargs = build_model_kvargs(args, rank_id)
        group_size = 2 if (args.enable_decode_microbatch_overlap or args.enable_prefill_microbatch_overlap) else 1
        if group_size == 2:
            for case in cases:
                assert case.batch_size % 2 == 0, "microbatch overlap requires even batch_size"

        init_distributed_env(model_kvargs)
        dist_group_manager.create_groups(group_size=group_size)
        model_cfg, _ = PretrainedConfig.get_config_dict(args.model_dir)
        dist.barrier()

        torch.cuda.empty_cache()
        model, _ = get_model(model_cfg, model_kvargs)
        cases = resolve_batch_max_prefill_cases(args, cases, model.mem_manager.size)
        cases = resolve_profile_decode_cases(args, cases, model.mem_manager.size)
        draft_models = init_mtp_draft_models(args, model_kvargs, model)
        token_source = TokenSource(args)
        executor = StaticBenchmarkExecutor(args, model, draft_models, token_source)

        results = []
        for case in cases:
            if args.warmup_iters > 0:
                executor.run_case(case, warmup=True)
            result = executor.run_case(case, warmup=False)
            if rank_id == 0:
                results.append(asdict(result))
            dist.barrier()

        ans_queue.put({"ok": True, "rank": rank_id, "results": results})
    except Exception:
        ans_queue.put({"ok": False, "rank": rank_id, "traceback": traceback.format_exc()})
    finally:
        try:
            ans_queue.close()
            ans_queue.join_thread()
        except Exception:
            pass
        os._exit(0)


def fmt_optional(value, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def print_aligned_table(headers: Sequence[str], rows: Sequence[Sequence[str]]):
    """Print a compact right-aligned ASCII table."""
    if not rows:
        return
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(str(value).rjust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers), flush=True)
    print("  ".join("-" * width for width in widths), flush=True)
    for row in rows:
        print(format_row(row), flush=True)


def prefill_table_row(result: BenchmarkResult) -> List[str]:
    """Format one prefill result row for stdout table output."""
    return [
        str(result.context_len),
        f"{result.cache_hit_rate:.2f}",
        str(result.batch_size),
        fmt_optional(result.profiled_max_total_token_num, 0),
        fmt_optional(result.prefill_uncached_len, 0),
        fmt_optional(result.prefill_cached_len, 0),
        str(result.measured_tokens),
        f"{result.elapsed_ms:.3f}",
        f"{result.qps:.2f}",
        f"{result.tps:.2f}",
        fmt_optional(result.logical_tps, 2),
    ]


def decode_table_row(result: BenchmarkResult) -> List[str]:
    """Format one decode result row for stdout table output."""
    return [
        str(result.context_len),
        str(result.batch_size),
        fmt_optional(result.mtp_accept_rate, 2),
        fmt_optional(result.profiled_max_total_token_num, 0),
        f"{result.elapsed_ms:.3f}",
        f"{result.qps:.2f}",
        f"{result.tps:.2f}",
        fmt_optional(result.inter_token_latency_ms, 3),
    ]


def print_results_table(results: Sequence[BenchmarkResult]):
    """Print separate prefill/decode tables for measured results."""
    prefill_rows = [prefill_table_row(result) for result in results if result.stage == "prefill"]
    decode_rows = [decode_table_row(result) for result in results if result.stage == "decode"]

    if prefill_rows:
        print("\n[prefill]", flush=True)
        print_aligned_table(PREFILL_TABLE_HEADERS, prefill_rows)
    if decode_rows:
        print("\n[decode]", flush=True)
        print_aligned_table(DECODE_TABLE_HEADERS, decode_rows)


def run_benchmark(args: SimpleNamespace, cases: Sequence[BenchmarkCase]) -> List[Dict]:
    ctx = mp.get_context("spawn")
    ans_queue = ctx.Queue()
    workers = []
    rank_start = args.node_rank * args.tp
    rank_end = (args.node_rank + 1) * args.tp
    case_dicts = [asdict(case) for case in cases]
    args_dict = vars(args)

    for rank_id in range(rank_start, rank_end):
        proc = ctx.Process(target=run_worker, args=(args_dict, case_dicts, rank_id, ans_queue))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    messages = []
    while not ans_queue.empty():
        messages.append(ans_queue.get())

    failed = [message for message in messages if not message.get("ok")]
    failed.extend(
        {
            "ok": False,
            "rank": index,
            "traceback": f"worker exited with code {proc.exitcode}",
        }
        for index, proc in enumerate(workers)
        if proc.exitcode not in (0, None)
    )
    if failed:
        for item in failed:
            print(
                f"rank {item.get('rank')} failed:\n{item.get('traceback')}",
                file=sys.stderr,
            )
        raise RuntimeError(f"{len(failed)} worker(s) failed")

    results = []
    for message in messages:
        results.extend(message.get("results") or [])
    result_objs = [BenchmarkResult(**result) for result in results]
    print_results_table(result_objs)
    return results


def add_static_benchmark_args(parser: argparse.ArgumentParser):
    parser.add_argument("--benchmark", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--batch_size", type=int, default=None, help="legacy single batch size")
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default=None,
        help="comma/space separated batch sizes",
    )
    parser.add_argument("--input_len", type=int, default=64, help="legacy single prefill/context length")
    parser.add_argument(
        "--input_lens",
        type=str,
        default=None,
        help="comma/space separated prefill lengths",
    )
    parser.add_argument(
        "--context_lens",
        type=str,
        default=None,
        help="comma/space separated decode context lengths",
    )
    parser.add_argument("--output_len", type=int, default=512, help="legacy single decode output length")
    parser.add_argument(
        "--output_lens",
        type=str,
        default=None,
        help="comma/space separated decode output lengths",
    )
    parser.add_argument(
        "--chunked_prefill_sizes",
        type=str,
        default=4096,
        help=("comma/space separated prefill chunk sizes; default is 4096 " "(full/none/0 select unchunked prefill)"),
    )
    parser.add_argument(
        "--prefill_cache_hit_rates",
        type=str,
        default=None,
        help=(
            "comma/space separated cache hit rates for prefill, e.g. "
            "'0,0.5,0.8,0.9'; uncached tokens are ceil(input_len * (1-hit))"
        ),
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=2048,
        help="upper bound for auto-computed prefill/decode batch size; <=0 disables it",
    )
    parser.add_argument(
        "--decode_batch_size_mode",
        choices=["explicit", "profile"],
        default="explicit",
        help=(
            "explicit uses --batch_size/--batch_sizes; profile computes decode BS "
            "from profiled max_total_token_num per context"
        ),
    )
    parser.add_argument(
        "--mtp_accept_rate",
        type=float,
        default=1.0,
        help=("per-draft-token MTP acceptance probability; sampling is outside " "the timed decode section"),
    )
    parser.add_argument("--warmup_iters", type=int, default=1)
    parser.add_argument("--bench_iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)


def main(argv: Optional[Sequence[str]] = None):
    parser = make_argument_parser()
    add_static_benchmark_args(parser)
    args = parser.parse_args(argv)
    if args.benchmark in {"all", "prefill"} and args.batch_max_tokens is None:
        args.batch_max_tokens = 8192
    cases = build_cases(args)
    if not cases:
        raise ValueError("no benchmark cases were generated")
    args = normalize_args(args, cases)
    set_env_start_args(args)

    run_benchmark(args, cases)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
