"""Collect speculative acceptance statistics for each request."""


class SpecAcceptanceMetrics:
    def __init__(self):
        self.bootstrap = {}
        self.proposals = {}
        self.verification_rounds = 0
        self.output_ranges = []
        self._bootstrap_next = True

    def mark_bootstrap(self):
        """Record the next verification round as initialization after prefill."""
        self._bootstrap_next = True

    def record(self, accept_length: int, verify_width: int, output_end=None):
        if not 1 <= accept_length <= verify_width:
            raise ValueError("accept length must include one target token and fit the verification width")
        # The first decode after prefill verifies the target-generated token.
        # Record it separately from subsequent draft proposals.
        bootstrap = self._bootstrap_next
        destination = self.bootstrap if bootstrap else self.proposals
        histogram = destination.setdefault(verify_width, [0] * verify_width)
        histogram[accept_length - 1] += 1
        if output_end is not None:
            self.output_ranges.append((bootstrap, verify_width, accept_length, output_end - accept_length + 1))
        self.verification_rounds += 1
        self._bootstrap_next = False

    @staticmethod
    def summarize(histograms):
        rounds = sum(sum(hist) for hist in histograms.values())
        accepted = sum(sum(n * count for n, count in enumerate(hist)) for hist in histograms.values())
        candidates = sum((width - 1) * sum(hist) for width, hist in histograms.items())
        positions = []
        for position in range(1, max(histograms, default=1)):
            eligible = [hist for width, hist in histograms.items() if width > position]
            proposed = sum(sum(hist) for hist in eligible)
            reached = sum(sum(hist[position - 1 :]) for hist in eligible)
            accepted_at_position = sum(sum(hist[position:]) for hist in eligible)
            positions.append(dict(position=position, proposed=proposed, reached=reached, accepted=accepted_at_position))
        return dict(
            rounds=rounds,
            accepted_draft_tokens=accepted,
            proposed_draft_tokens=candidates,
            histogram_by_verify_width={str(w): list(h) for w, h in sorted(histograms.items())},
            positions=positions,
        )

    def export(self, native_final_counters=None, visible_output_tokens=None, inferred_output_tokens=None):
        result = dict(
            schema_version=2,
            scope="verified_prefix_before_eos_or_output_budget_truncation",
            bootstrap=self.summarize(self.bootstrap),
            proposals=self.summarize(self.proposals),
        )
        if native_final_counters is not None:
            result["native_final_counters"] = dict(native_final_counters)
        if visible_output_tokens is not None:
            result["visible_output_tokens"] = visible_output_tokens
            result["inferred_output_tokens"] = inferred_output_tokens
            committed = dict(proposal_rounds=0, output_tokens=0, accepted_draft_tokens=0, unused_proposal_rounds=0)
            for bootstrap, width, length, start in self.output_ranges:
                if bootstrap:
                    continue
                visible = max(0, min(length, visible_output_tokens - start + 1))
                if visible:
                    committed["proposal_rounds"] += 1
                    committed["output_tokens"] += visible
                    committed["accepted_draft_tokens"] += min(length - 1, visible)
                else:
                    committed["unused_proposal_rounds"] += 1
            result["committed"] = committed
            result["verification_output_ranges"] = [list(row) for row in self.output_ranges]
        return result
