"""Contians policy-related implementations.

Includes Policy base class.
"""

from RichmanRL.utils import ValueFunction
from RichmanRL.envs.typing_utils import RichmanObservation, RichmanAction
from abc import ABC, abstractmethod
import numpy as np
from .mlp import MLP
import torch.optim as optim
from typing import Union, Literal
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
        state: RichmanObservation,
    ) -> int:
        """All policies must be callable, returning an action."""
        # Generics would be a better way to type hint this.

    @abstractmethod
    def update(self, *args, **kwargs):
        """All policies must have an update method.

        This can be updating the weights of a neural network in polciy
        gradient, or as simple as making a policy greedy with respect
        to an updated value function. If a value function is required
        for the policy, it should be updated here as well.
        """
        pass


class NoBiddingPolicy(Policy):
    """Useful for learning game policies independently."""

    def __call__(self, state: RichmanObservation) -> int:
        """Always returns the same number (0).

        Ties on bidding passed into the RichmanEnv will result in a
        50/50 split on who gets the action.
        """
        return 0

    def update(self, *args, **kwargs):
        """Does nothing."""
        pass


class RandomBiddingPolicy(Policy):
    """Fully random policy."""

    def __call__(
        self,
        state: RichmanObservation,
    ) -> int:
        """For a random policy, just returns a purely random (legal) action."""
        highest_bid = state["action_mask"][0]
        return np.random.randint(highest_bid + 1)

    def update(self, *args, **kwargs):
        """For a random policy, update does nothing."""
        pass


class RandomGamePolicy(Policy):
    """Fully random policy for a game."""

    def __call__(self, state: RichmanObservation) -> int:
        """For a random policy, just return a purely random (legal) action."""
        legal_mask = state["action_mask"][1]
        legal_probs = self.probs * legal_mask

        legal_indices = [i for i in range(len(legal_probs)) if legal_probs[i] > 0]

        index = np.random.choice(legal_indices)

        return index

    def update(self, *args, **kwargs):
        """Does nothing."""
        pass


class InGameNNPolicy(Policy):
    """Approximation of a policy with a neural network."""

    def __init__(self, input_dimension: int, output_dimension: int, alpha: float):
        """Constructor.

        Args:
            input_dimension: Dimension of the flattened state space.
            output_dimension: Number of actions
            alpha: For updating with REINFORCE
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.alpha = alpha

        self.mlp = MLP(input_dimension, output_dimension)

        self.optimizer = optim.Adam(
            self.mlp.parameters(), lr=0.0003, betas=(0.9, 0.999)
        )

    @torch.enable_grad()
    def update(
        self,
        state: RichmanObservation,
        action: RichmanAction,
        gamma_t: float,
        delta: float,
        agent: Union[Literal["player_1", "player_2"]],
    ):
        """One update step of policy gradient algorithm."""
        if not action:
            #The last step in trajectory has no valid action.
            return
        
        #print("\n")
        self.mlp.train()
        self.optimizer.zero_grad()

        state_feature = state["observation"][2].flatten()
        legal_mask = state["action_mask"][1]

        action_taken = action[agent][1]

        probs = self.mlp(state_feature, legal_mask, return_prob=True)

        #print(f"probs is {probs} and action taken is {action_taken}")

        loss = -1 * torch.log(probs[action_taken]) * self.alpha * gamma_t * delta

        #print(f"Loss is {loss}")

        loss.backward()
        self.optimizer.step()

    def __call__(self, state: RichmanObservation) -> int:
        """Callable that returns action for the agent."""
        state_feature = state["observation"][2].flatten()
        legal_mask = state["action_mask"][1]

        action_probs = self.mlp(state_feature, legal_mask, return_prob=True)

        _, action = torch.max(action_probs, dim=0)

        #print(f"Legal_mask is {legal_mask}")
        #print(f"Action_probs is {action_probs}")
        #print(f"Action is {action}")

        return action
