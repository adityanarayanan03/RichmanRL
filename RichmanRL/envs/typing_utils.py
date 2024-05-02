"""Contains definitions used exclusively for type checking and type hinting."""
from typing import Tuple, TypedDict, Any


class RichmanAction(TypedDict):
    """Used for typig an action provided to a Richman Game env."""
    player_1: Tuple[int, int]
    player_2: Tuple[int, int]

class RichmanObservation(TypedDict):
    """Used for typing observations of a richman environment."""
    observation: Tuple[int, int, Any]
    action_mask: Tuple[int, Any]