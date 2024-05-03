"""Environments for Richman games.

These environments are wrappers of pettingzoo environments. The wrappers take 
care of assessing bids, augmenting state spaces with bidding, augmenting
observation and reward structure with money, and switching turns according to
pricing.
"""

from .richman_env import RichmanEnv   # noqa: F401
from .HexEnv import Hex, HexBoard