"""Shared type aliases for EPLB placement planning."""

from typing import List, Tuple


# [rank][local physical expert] -> logical expert
LayerPlacement = List[List[int]]
# [layer][rank][local physical expert] -> logical expert
ExpertPlacement = List[LayerPlacement]
# [logical expert][replica metadata]
LogicalToPhysicalMap = List[List[int]]
# (logical expert, replica count, aligned load per replica)
ExpertReplicaGroup = Tuple[int, int, float]
