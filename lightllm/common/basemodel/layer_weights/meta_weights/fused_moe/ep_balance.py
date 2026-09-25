from dataclasses import dataclass


@dataclass(slots=True)
class PrefillEPBalanceCounters:
    """Cumulative CPU loads for one EP MoE layer's completed prefill dispatches."""

    route_load: int = 0
    compute_load: int = 0

    def accumulate(self, route_load: int, compute_load: int):
        """Accumulate exact route and alignment-expanded compute loads for one prefill dispatch."""
        self.route_load += route_load
        self.compute_load += compute_load
