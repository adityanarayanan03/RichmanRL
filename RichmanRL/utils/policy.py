"""Contians policy-related implementations.

Includes Policy base class.
"""

from RichmanRL.utils import ValueFunction
from RichmanRL.envs.typing_utils import RichmanObservation
from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    """Abstract base class for policies, tabular or NN."""

    def __init__(self, value_function: ValueFunction, num_actions, epsilon=0.0):
        """Constructor.

        Args:
            value_function : ValueFunction - Valid state or state-action value function.
            num_actions : Number of actions available.
            epsilon : float - Epsilon value for exploration. Set to 0 for none.
        """
        self.value_function = value_function
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.probs = np.array([1 / self.num_actions for _ in range(self.num_actions)])

    @abstractmethod
    def __call__(
        self,
        state: RichmanObservation,
    ) -> int:
        """All policies must be callable, returning an action."""
        # Generics would be a better way to type hint this.

    @abstractmethod
    def update(self,  *args, **kwargs):
        """All policies must have an update method.
        
        This can be updating the weights of a neural network in polciy
        gradient, or as simple as making a policy greedy with respect
        to an updated value function. If a value function is required
        for the policy, it should be updated here as well.
        """
        pass


class RandomBiddingPolicy(Policy):
    """Fully random policy."""

    def __call__(
        self,
        state: RichmanObservation,
    ) -> int:
        """For a random policy, just returns a purely random (legal) action."""
        highest_bid = state['action_mask'][0]
        legal_bids = [1 if i <= highest_bid else 0 for i in range(self.num_actions)]
        legal_probs = self.probs * legal_bids
        return np.argmax(legal_probs)

    def update(self, *args, **kwargs):
        """For a random policy, update does nothing."""
        pass

class RandomGamePolicy(Policy):
    """Fully random policy for a game."""

    def __call__(self, state: RichmanObservation) -> int:
        """For a random policy, just return a purely random (legal) action."""
        legal_mask = state['action_mask'][1]
        legal_probs = self.probs * legal_mask
        return np.argmax(legal_probs)
    
    def update(self, *args, **kwargs):
        """Does nothing."""
        pass