"""DeepEP dispatch/combine over buffers bound at startup."""

from dataclasses import dataclass

from lightllm.common.execution.transfer.completion import TransferCompletion


class ExpertFence:
    def __init__(self, receive, driver, keepalive):
        self.receive, self.driver, self.keepalive = receive, driver, keepalive
        self.event = None

    def wait(self):
        if self.event is None:
            if self.driver.failure is not None:
                raise RuntimeError(
                    "EP transfer is quarantined after an incomplete submission"
                ) from self.driver.failure[0]
            try:
                self.receive()
                self.event = self.driver.runtime.record_event()
            except Exception as exc:
                self.driver.failure = (exc, self.keepalive)
                raise

    def query(self):
        return self.event is not None and self.event.query()

    def synchronize(self):
        self.wait()
        self.event.synchronize()


@dataclass
class ExpertDispatch:
    inputs: object
    indices: object
    weights: object
    counts: object
    handle: object
    completion: TransferCompletion

    @property
    def num_unaligned_recv_tokens_per_expert(self):
        return self.handle.num_unaligned_recv_tokens_per_expert

    @property
    def recv_src_metadata(self):
        return self.handle.recv_src_metadata

    def wait(self):
        self.completion.destination_ready.wait()


class ExpertTransfer:
    def __init__(
        self,
        prefill_buffer,
        decode_buffer,
        *,
        experts,
        prefill_limit,
        decode_limit,
        num_sms,
        round_scales,
        separate_contributions=False,
        runtime=None,
        capture=None,
    ):
        if prefill_buffer is None or decode_buffer is None:
            raise ValueError("EP buffers must exist before binding execution")
        if min(experts, prefill_limit, decode_limit) <= 0:
            raise ValueError("EP transfer limits must be positive")
        for obj, names in (
            (prefill_buffer, ("dispatch", "combine")),
            (decode_buffer, ("low_latency_dispatch", "low_latency_combine")),
        ):
            if any(not callable(getattr(obj, name, None)) for name in names):
                raise ValueError("EP buffer lacks the selected transfer operations")
        if runtime is None:
            from lightllm.common.execution.device.cuda import CudaRuntime

            runtime = CudaRuntime()
        if capture is None:
            from deep_ep import ElasticBuffer

            capture = ElasticBuffer.capture
        self.prefill_buffer, self.decode_buffer = prefill_buffer, decode_buffer
        self.experts, self.prefill_limit, self.decode_limit = experts, prefill_limit, decode_limit
        self.num_sms, self.round_scales, self.runtime, self.capture = num_sms, round_scales, runtime, capture
        self.separate_contributions = separate_contributions
        self.failure = None

    def _completion(self, receive, keepalive):
        fence = ExpertFence(receive, self, keepalive)
        return TransferCompletion(fence, fence, keepalive)

    def dispatch(self, inputs, indices, weights, *, prefill, previous_event=None, overlap=False):
        if indices.ndim != 2 or weights.shape != indices.shape:
            raise ValueError("EP dispatch needs matching token/top-k indices and weights")
        limit = self.prefill_limit if prefill else self.decode_limit
        if indices.shape[0] > limit:
            raise ValueError("EP token count exceeds the bound transfer capacity")
        if self.failure is not None:
            raise RuntimeError("EP transfer is quarantined after an incomplete submission") from self.failure[0]
        try:
            return self._dispatch(inputs, indices, weights, prefill, previous_event, overlap)
        except Exception as exc:
            # The communication stream may still own these tensors. A local error
            # alone cannot establish that every rank has stopped accessing them.
            self.failure = (exc, inputs, indices, weights)
            raise

    def _dispatch(self, inputs, indices, weights, prefill, previous_event, overlap):
        if prefill:
            x, ids, scores, handle, event = self.prefill_buffer.dispatch(
                inputs,
                topk_idx=indices,
                topk_weights=weights,
                num_experts=self.experts,
                num_max_tokens_per_rank=self.prefill_limit,
                expert_alignment=128,
                num_sms=self.num_sms,
                previous_event=previous_event,
                async_with_compute_stream=overlap,
                allocate_on_comm_stream=overlap,
                do_cpu_sync=True,
                do_handle_copy=False,
                do_expand=True,
                use_tma_aligned_col_major_sf=True,
            )
            counts = handle.num_recv_tokens_per_expert_list
            receive = event.current_stream_wait if overlap else lambda: None
        else:
            x, counts, handle, event, hook = self.decode_buffer.low_latency_dispatch(
                inputs,
                indices,
                self.decode_limit,
                self.experts,
                use_fp8=True,
                round_scale=self.round_scales,
                async_finish=False,
                return_recv_hook=overlap,
            )
            ids, scores = indices, weights
            receive = hook if overlap else lambda: None
        completion = self._completion(receive, (inputs, indices, weights, x, ids, scores, counts, handle, event))
        return ExpertDispatch(x, ids, scores, counts, handle, completion)

    def combine(self, inputs, exchange, *, prefill, previous_event=None, overlap=False):
        if self.failure is not None:
            raise RuntimeError("EP transfer is quarantined after an incomplete submission") from self.failure[0]
        try:
            if self.separate_contributions:
                return self._combine_contributions(inputs, exchange, prefill, previous_event)
            return self._combine(inputs, exchange, prefill, previous_event, overlap)
        except Exception as exc:
            if self.failure is None:
                self.failure = (exc, inputs, exchange)
            raise

    def _combine_contributions(self, inputs, exchange, prefill, previous_event):
        import torch

        count = exchange.recv_src_metadata.shape[1] - 2 if prefill else exchange.indices.shape[1]
        if prefill and (inputs.ndim != 3 or inputs.shape[0] != count):
            raise ValueError("EP combine requires one receive tensor per top-k slot")
        outputs, keepalive = [], [inputs, exchange]
        try:
            for slot in range(count):
                # Wait on each communication submission before reusing DeepEP's
                # internal slots. These waits are stream dependencies, not CPU syncs.
                if prefill:
                    output, _, event = self.prefill_buffer.combine(
                        inputs[slot],
                        exchange.handle,
                        topk_weights=None,
                        num_sms=self.num_sms,
                        previous_event=previous_event if slot == 0 else None,
                        async_with_compute_stream=False,
                        allocate_on_comm_stream=previous_event is not None and slot == 0,
                    )
                else:
                    weights = torch.zeros_like(exchange.weights)
                    weights[:, slot].copy_(exchange.weights[:, slot])
                    keepalive.append(weights)
                    output, event, _ = self.decode_buffer.low_latency_combine(
                        inputs,
                        exchange.indices,
                        weights,
                        exchange.handle,
                        async_finish=False,
                        return_recv_hook=False,
                    )
                outputs.append(output)
                keepalive.extend((output, event))
        except Exception as exc:
            self.failure = (exc, keepalive)
            raise
        return tuple(outputs), self._completion(lambda: None, keepalive)

    def _combine(self, inputs, exchange, prefill, previous_event, overlap):
        if prefill:
            output, _, event = self.prefill_buffer.combine(
                inputs,
                exchange.handle,
                topk_weights=None,
                num_sms=self.num_sms,
                previous_event=previous_event,
                async_with_compute_stream=overlap,
                allocate_on_comm_stream=overlap,
            )
            receive = event.current_stream_wait if overlap else lambda: None
        else:
            output, event, hook = self.decode_buffer.low_latency_combine(
                inputs,
                exchange.indices,
                exchange.weights,
                exchange.handle,
                async_finish=False,
                return_recv_hook=overlap,
            )
            receive = hook if overlap else lambda: None
        return output, self._completion(receive, (inputs, exchange, output, event))
