"""Collection of typing utilities."""
from typing import TypedDict, List, Tuple
from RichmanRL.envs.typing_utils import RichmanObservation, RichmanAction

class AgentTrajectory(TypedDict):
    """Represents the trajectories viewed by two agents."""
    player_1: List[Tuple[float, RichmanObservation, RichmanAction]]
    player_2: List[Tuple[float, RichmanObservation, RichmanAction]]