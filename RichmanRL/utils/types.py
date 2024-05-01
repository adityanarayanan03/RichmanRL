"""Collection of typing utilities."""
from __future__ import annotations
from typing import TypedDict

class AgentTrajectory(TypedDict):
    """Represents the trajectories viewed by two agents."""
    player_1: list[tuple[float, RichmanObservation, RichmanAction]]
    player_2: list[tuple[float, RichmanObservation, RichmanAction]]