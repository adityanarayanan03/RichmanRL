"""Contains definitions used exclusively for type checking and type hinting."""
from typing import Tuple, TypedDict

class RichmanAction(TypedDict):
    """Used for typig an action provided to a Richman Game env."""
    player_1: Tuple[int, int]
    player_2: Tuple[int, int]