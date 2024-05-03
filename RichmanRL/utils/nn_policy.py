"""Neural Network policies for policy gradient."""
from __future__ import annotations
from .policy import Policy
import torch
import torch.optim as optim
from typing import Union, Literal
from .mlp import MLP
import numpy as np
import sys
import logging


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
        agent: Literal["player_1"] | Literal["player_2"],
    ):
        """One update step of policy gradient algorithm."""
        if not action:
            # The last step in trajectory has no valid action.
            return

        self.mlp.train()
        self.optimizer.zero_grad()

        state_feature = state["observation"][2].flatten()
        legal_mask = state["action_mask"][1]

        action_taken = action[agent][1]

        probs = self.mlp(state_feature, legal_mask, return_prob=True)

        loss = -1 * torch.log(probs[action_taken]) * self.alpha * gamma_t * delta

        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def __call__(self, state: RichmanObservation, return_probs = False) -> int:
        """Callable that returns action for the agent."""
        state_feature = state["observation"][2].flatten()
        legal_mask = state["action_mask"][1]

        action_probs = self.mlp(state_feature, legal_mask, return_prob=True)

        _, action = torch.max(action_probs, dim=0)

        if return_probs:
            return action_probs.numpy(), int(action)

        return int(action)


class BiddingNNPolicy(Policy):
    """Approximation of a bidding policy with a neural network."""

    def __init__(
        self, input_dimension: int, output_dimension: int, alpha: float, verbose=True
    ):
        """Constructor.

        Args:
            input_dimension: Should the the bidding dimension + the game state
            output_dimension: For discrete bidding.
            alpha: For updating with REINFORCE
            verbose: Whether or not to enable logging statements
        """
        self.logger = logging.getLogger("BiddingNNPolicy")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)

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
        agent=Union[Literal["player_1", "player_2"]],
    ):
        """One update step of policy gradient algorithm."""
        if not action:
            # The last step has no valid action.
            return

        self.mlp.train()
        self.optimizer.zero_grad()

        bidding_feature = state["observation"][0]  # My own pot size
        state_feature = state["observation"][2].flatten()  # Game state
        input_feature = np.concatenate([[bidding_feature], state_feature])

        legal_bid = state["action_mask"][0]
        legal_mask = [1 if i <= legal_bid else 0 for i in range(self.output_dimension)]

        action_taken = action[agent][0]

        if legal_mask[action_taken] == 0:
            self.logger.error(f"Agent {agent} took an illegal action {action}")
            self.logger.debug(f"Legal bid was {legal_bid} and action was {action_taken}")  # noqa: E501
            sys.exit(0)

        probs = self.mlp(input_feature, legal_mask, return_prob=True)

        loss = -1 * torch.log(probs[action_taken]) * self.alpha * gamma_t * delta

        loss.backward()
        self.optimizer.step()

    @torch.no_grad
    def __call__(self, state: RichmanObservation, return_probs = False) -> int:
        """Callable that returns action for the agent."""
        bidding_feature = state["observation"][0]  # My own pot size
        state_feature = state["observation"][2].flatten()  # Game state
        input_feature = np.concatenate([[bidding_feature], state_feature])

        legal_bid = state["action_mask"][0]
        legal_mask = [1 if i <= legal_bid else 0 for i in range(self.output_dimension)]

        action_probs = self.mlp(input_feature, legal_mask, return_prob=True)

        _, action = torch.max(action_probs, dim=0)

        if return_probs:
            return action_probs.numpy(), int(action)

        return int(action)
