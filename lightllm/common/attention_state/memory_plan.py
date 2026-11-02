from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MemoryDemand:
    name: str
    domain: str
    fixed_bytes: int = 0
    bytes_per_unit: int = 0
    tokens_per_unit: int = 1

    def __post_init__(self):
        if self.fixed_bytes < 0 or self.bytes_per_unit < 0 or self.tokens_per_unit <= 0:
            raise ValueError("memory requirements must be nonnegative and have positive units")

    def at_capacity(self, capacity):
        return self.fixed_bytes + (capacity + self.tokens_per_unit - 1) // self.tokens_per_unit * self.bytes_per_unit


@dataclass(frozen=True)
class MemoryPlan:
    capacity: int
    demands: tuple[MemoryDemand, ...]
    limits: tuple[tuple[str, int], ...]

    def usage(self, domain):
        return sum(demand.at_capacity(self.capacity) for demand in self.demands if demand.domain == domain)

    def with_runtime_peak(self, peak_bytes, domain="cuda"):
        reserve = sum(d.fixed_bytes for d in self.demands if d.name == "runtime_headroom" and d.domain == domain)
        demands = tuple(d for d in self.demands if d.name != "runtime_headroom")
        demands += (
            MemoryDemand("observed_runtime_peak", domain, peak_bytes),
            MemoryDemand("runtime_margin", domain, max(0, reserve - peak_bytes)),
        )
        return replace(self, demands=demands)

    def explain(self):
        return {
            domain: dict(
                limit_bytes=limit,
                planned_bytes=self.usage(domain),
                allocations={d.name: d.at_capacity(self.capacity) for d in self.demands if d.domain == domain},
            )
            for domain, limit in self.limits
        }


def merge_memory_demands(*groups):
    allocations = {}
    for group in groups:
        for demand in group:
            previous = allocations.setdefault(demand.name, demand)
            if previous != demand:
                raise ValueError(f"inconsistent requirements for shared allocation {demand.name}")
    return tuple(allocations.values())


def plan_memory(demands, limits, page_size, requested_capacity=None):

    demands = tuple(demands)
    if page_size <= 0 or any(value < 0 for value in limits.values()):
        raise ValueError("invalid page size or memory limit")
    if len({d.name for d in demands}) != len(demands):
        raise ValueError("memory allocations must be counted once, with unique names")
    if any(d.domain not in limits for d in demands):
        raise ValueError("a memory domain has no budget")

    def exceeds(capacity):
        for domain, limit in limits.items():
            required = sum(d.at_capacity(capacity) for d in demands if d.domain == domain)
            if required > limit:
                return domain, required, limit
        return None

    if requested_capacity is not None:
        capacity = requested_capacity // page_size * page_size
        overflow = exceeds(capacity)
    else:
        overflow = exceeds(page_size)
        if overflow is None:
            bounds = [
                limits[d.domain] // d.bytes_per_unit * d.tokens_per_unit // page_size
                for d in demands
                if d.bytes_per_unit
            ]
            if not bounds:
                raise ValueError("capacity requires a growing resource or an explicit limit")
            low, high = 1, min(bounds)
            while low < high:
                mid = (low + high + 1) // 2
                if exceeds(mid * page_size):
                    high = mid - 1
                else:
                    low = mid
            capacity = low * page_size
    if overflow is not None:
        domain, required, limit = overflow
        raise MemoryError(
            f"{domain} budget cannot fit attention storage including HOLD/reserved pages: "
            f"requires {required} bytes, limit {limit} bytes"
        )
    if capacity <= 0:
        raise ValueError("attention capacity must contain at least one complete page")
    return MemoryPlan(capacity, demands, tuple(limits.items()))
