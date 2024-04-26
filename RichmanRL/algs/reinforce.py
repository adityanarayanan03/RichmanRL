"""Implements the REINFORCE algorithm."""

from RichmanRL.envs import RichmanEnv
from RichmanRL.utils import Policy
from itertools import count
from typing import List, Tuple
from RichmanRL.envs.typing_utils import RichmanObservation, RichmanAction


def generate_trajectory(
    env: RichmanEnv, pi : Policy
) -> List[Tuple[float, RichmanObservation, RichmanAction]]:
    """Generate a complete trajectory.

    Args:
        env : RichmanEnv - An environment to generate trajectory from
        pi : Policy - A policy to sample from.
    """
    for t in count():
        pass
