"""Contians policy-related implementations.

Includes Policy base class.
"""

from RichmanRL.utils import ValueFunction
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch


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
        features: Union[np.ndarray, torch.Tensor],
        legal_mask: Union[np.ndarray, torch.Tensor],
    ) -> int:
        """All policies must be callable, returning an action."""
        # Generics would be a better way to type hint this.


class RandomPolicy(Policy):
    """Fully random policy."""

    def __call__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        legal_mask: Union[np.ndarray, torch.Tensor],
    ) -> int:
        """For a random policy, just returns a purely random (legal) action."""
        legal_probs = self.probs * legal_mask
        return np.argmax(legal_probs)
