from .pd_selector import PDSelector, RandomSelector, RoundRobinSelector, MemorySelector


def create_selector(selector_type: str, pd_manager) -> PDSelector:
    if selector_type == "random":
        return RandomSelector(pd_manager)
    elif selector_type == "round_robin":
        return RoundRobinSelector(pd_manager)
    elif selector_type == "memory":
        return MemorySelector(pd_manager)
    else:
        raise ValueError(f"Invalid selector type: {selector_type}")
