"""Startup execution and ownership of temporary warmup resources."""

import gc
import os
from contextlib import contextmanager

import torch
from lightllm.utils.progress import ThreadProgress

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.utils.log_utils import init_logger
from lightllm.core.device.cuda import CudaRuntime

logger = init_logger(__name__)


class WarmupCleanupError(RuntimeError):
    pass


class WarmupRunner:
    def __init__(
        self,
        requests,
        kv_allocator,
        forward,
        special_inputs,
        vocab_size,
        *,
        overlap_prefill,
        overlap_decode,
        runtime=CudaRuntime,
    ):
        self.requests = requests
        self.kv_allocator = kv_allocator
        self.forward = forward
        self.special_inputs = special_inputs
        self.vocab_size = vocab_size
        self.overlap_prefill = overlap_prefill
        self.overlap_decode = overlap_decode
        self.runtime = runtime
        self.device = requests.req_to_token_indexs.device

    def prefill_input(self, tokens, request_id, *, random=False):
        if tokens <= 0:
            raise ValueError("warmup requires a positive token count")
        if random:
            generator = torch.Generator(device=self.device).manual_seed(tokens)
            ids = torch.randint(self.vocab_size, (tokens,), dtype=torch.int64, device=self.device, generator=generator)
        else:
            ids = torch.ones(tokens, dtype=torch.int64, device=self.device)
        return ModelInput(
            batch_size=1,
            total_token_num=tokens,
            max_q_seq_len=tokens,
            max_kv_seq_len=tokens,
            max_cache_len=0,
            input_ids=ids,
            b_req_idx=torch.tensor([request_id], dtype=torch.int32, device=self.device),
            b_mtp_index=torch.zeros(1, dtype=torch.int32, device=self.device),
            b_seq_len=torch.full((1,), tokens, dtype=torch.int32, device=self.device),
            b_ready_cache_len=torch.zeros(1, dtype=torch.int32, device=self.device),
            b_prefill_start_loc=torch.zeros(1, dtype=torch.int32, device=self.device),
            b_is_decode_req=torch.zeros(1, dtype=torch.bool, device=self.device),
            is_prefill=True,
            b_prefill_has_output_cpu=[False],
            multimodal_params=[{"images": [], "audios": []}],
            **self.special_inputs(tokens),
        )

    def decode_input(self, rows, max_kv_length):
        return ModelInput(
            batch_size=rows,
            total_token_num=rows * 2,
            max_q_seq_len=1,
            max_kv_seq_len=max_kv_length,
            input_ids=torch.ones(rows, dtype=torch.int64, device=self.device),
            b_req_idx=torch.full((rows,), self.requests.HOLD_REQUEST_ID, dtype=torch.int32, device=self.device),
            b_seq_len=torch.full((rows,), 2, dtype=torch.int32, device=self.device),
            b_mtp_index=torch.zeros(rows, dtype=torch.int32, device=self.device),
            b_shared_seq_len=torch.zeros(rows, dtype=torch.int32, device=self.device),
            b_shared_radix_node_id=torch.full((rows,), -1, dtype=torch.int64, device=self.device),
            b_position_delta=torch.zeros(rows, dtype=torch.int32, device=self.device),
            is_prefill=False,
            multimodal_params=[{"images": [], "audios": []} for _ in range(rows)],
            **self.special_inputs(rows),
        )

    @contextmanager
    def allocated_prefill(self, tokens, *, random=False):
        request_id = self.requests.alloc()
        if request_id is None:
            raise RuntimeError("no request slot available for warmup")
        indices = None
        try:
            page = self.kv_allocator.page_size
            indices = self.kv_allocator.alloc((tokens + page - 1) // page * page)
            indices = indices.clone()
            self.requests.req_to_token_indexs[request_id, : len(indices)] = indices.to(self.device)
            yield self.prefill_input(tokens, request_id, random=random)
        finally:
            try:
                # A failed forward can still have work in flight on a communication stream.
                self.runtime.synchronize()
            except BaseException as error:
                raise WarmupCleanupError("warmup completion unknown; request and KV allocation retained") from error
            try:
                if indices is not None:
                    self.kv_allocator.free(indices)
                self.requests.free_req(request_id)
            except Exception as error:
                raise WarmupCleanupError("warmup resource release failed") from error

    @torch.no_grad()
    def capture_decode(self, graph, *, overlap=False):
        logger.info("Begin capture decode cudagraph")
        for size in reversed(graph.cuda_graph_batch_sizes):
            inputs = [self.decode_input(size, graph.graph_max_len_in_batch) for _ in range(2 if overlap else 1)]
            if overlap:
                self.overlap_decode(*inputs)
            else:
                self.forward(inputs[0])
            del inputs
            self.runtime.empty_cache()
        logger.info(f"Captured decode cudagraph through batch size {graph.max_batch_size}")

    @torch.no_grad()
    def capture_prefill(self, graph, *, overlap=False):
        logger.info("Begin capture prefill cudagraph")
        for size in reversed(graph.graph_handle_token_nums):
            inputs = [self.prefill_input(size, self.requests.HOLD_REQUEST_ID) for _ in range(2 if overlap else 1)]
            if overlap:
                self.overlap_prefill(*inputs)
            else:
                self.forward(inputs[0])
            del inputs
            self.runtime.empty_cache()
        logger.info(f"Captured prefill cudagraph through {graph.max_handle_token_num} tokens")

    @torch.no_grad()
    def initialize_hold(self):
        self.runtime.barrier()
        self.forward(self.prefill_input(1, self.requests.HOLD_REQUEST_ID))
        self.runtime.empty_cache()

    @torch.no_grad()
    def autotune(self, max_tokens, recipe_scope):
        lengths = sorted({1, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, max_tokens}, reverse=True)
        self.runtime.barrier()
        with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL), recipe_scope():
            for length in ThreadProgress([length for length in lengths if length <= max_tokens], desc="warming up"):
                try:
                    with self.allocated_prefill(length, random=True) as batch:
                        self.forward(batch)
                except WarmupCleanupError:
                    raise
                except Exception:
                    logger.exception(f"autotune warmup for length {length} failed")
                finally:
                    gc.collect()
                    self.runtime.empty_cache()
        self.runtime.barrier()

    @torch.no_grad()
    def check_max_length(self, tokens):
        if os.getenv("DISABLE_CHECK_MAX_LEN_INFER") is not None:
            logger.info("disable_check_max_len_infer is true")
            return
        self.runtime.barrier()
        try:
            with self.allocated_prefill(tokens) as batch:
                output = self.forward(batch)
                torch.argmax(torch.softmax(output.logits, dim=-1), dim=1, keepdim=True)
                del output
        except WarmupCleanupError:
            raise
        except (RuntimeError, torch.OutOfMemoryError) as error:
            raise RuntimeError(
                "check max len infer fail; reduce --mem_fraction/--max_total_token_num, "
                "--max_req_total_len or --batch_max_tokens"
            ) from error
        finally:
            self.runtime.empty_cache()
        logger.info(f"check max_len {tokens} infer ok")
