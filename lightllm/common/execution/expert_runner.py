"""Order routed expert math and transfer without coupling the two backends."""


class ExpertRunner:
    def __init__(self, compute, transfer):
        if compute.total_experts != transfer.experts or compute.round_scales != transfer.round_scales:
            raise ValueError("EP compute and transfer disagree on placement or encoding")
        if compute.spec.round_intermediates != transfer.separate_contributions:
            raise ValueError("EP compute and transfer disagree on contribution rounding")
        self.compute, self.transfer = compute, transfer
        self.capture = transfer.capture

    def run(self, inputs, logits, *, is_prefill, lane=0, capture=None):
        spec = self.compute.spec
        if inputs.ndim != 2 or inputs.shape != (logits.shape[0], spec.hidden_size) or inputs.dtype != spec.dtype:
            raise ValueError("EP input disagrees with the bound model semantics")
        weights, ids = self.compute.route(logits, capture=capture)
        source = self.compute.quantize(inputs) if is_prefill else inputs
        exchange = self.transfer.dispatch(source, ids, weights, prefill=is_prefill)
        exchange.wait()
        if is_prefill:
            result = self.compute.expanded(
                exchange.counts,
                exchange.num_unaligned_recv_tokens_per_expert,
                exchange.recv_src_metadata,
                exchange.inputs,
                exchange.weights,
                lane,
            )
        else:
            result = self.compute.masked(exchange.inputs, exchange.counts, self.expected_m(inputs.shape[0]))
        output, ready = self._combine(result, exchange, prefill=is_prefill)
        ready()
        return output

    def expected_m(self, tokens):
        count = tokens * self.compute.world_size * self.compute.spec.top_k
        return (count + self.compute.total_experts - 1) // self.compute.total_experts

    # Model recipes use these steps to interleave two microbatches with attention.
    def select_experts_and_quant_input(self, inputs, logits, *, capture=None):
        weights, ids = self.compute.route(logits, capture=capture)
        return weights, ids, self.compute.quantize(inputs)

    def low_latency_dispatch(self, inputs, logits, *, capture=None):
        weights, ids = self.compute.route(logits, capture=capture)
        exchange = self.transfer.dispatch(inputs, ids, weights, prefill=False, overlap=True)
        return exchange.inputs, exchange.counts, ids, weights, exchange, exchange.wait

    def dispatch(self, inputs, ids, weights, overlap_event=None):
        exchange = self.transfer.dispatch(
            inputs, ids, weights, prefill=True, previous_event=overlap_event, overlap=True
        )
        return exchange.inputs, exchange.indices, exchange.weights, exchange.counts, exchange, exchange.wait

    def masked_group_gemm(self, inputs, counts, dtype, expected_m):
        if dtype != self.compute.spec.dtype:
            raise ValueError("EP decode dtype changed after binding")
        return self.compute.masked(inputs, counts, expected_m)

    def prefilled_group_gemm(
        self, counts, unaligned_counts, source_metadata, inputs, ids, weights, hidden_dtype=None, microbatch_index=0
    ):
        if ids is not None or hidden_dtype not in (None, self.compute.spec.dtype):
            raise ValueError("EP prefill requires expanded dispatch with the bound dtype")
        return self.compute.expanded(counts, unaligned_counts, source_metadata, inputs, weights, microbatch_index)

    def low_latency_combine(self, inputs, ids, weights, exchange):
        return self._combine(inputs, exchange, prefill=False, overlap=True)

    def combine(self, inputs, exchange, overlap_event=None):
        return self._combine(inputs, exchange, prefill=True, previous_event=overlap_event, overlap=True)

    def _combine(self, inputs, exchange, **kwargs):
        output, completion = self.transfer.combine(inputs, exchange, **kwargs)
        if not self.compute.spec.round_intermediates:
            return output, completion.destination_ready.wait
        completion.destination_ready.wait()
        output = self.compute.reduce(output)
        event = self.transfer.runtime.record_event()

        def ready():
            completion.destination_ready.wait()
            event.wait()

        return output, ready
