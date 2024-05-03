"""Contians policy-related implementations.

Includes Policy base class.
"""
from __future__ import annotations
from RichmanRL.utils import ValueFunction
from abc import ABC, abstractmethod
import numpy as np
import pickle
import os


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

class ConservativeBiddingPolicy(Policy):
    
    def __call__(
        self,
        state: RichmanObservation,
    ) -> int:
        """For a random policy, just returns a purely random (legal) action."""
        highest_bid = int(state["action_mask"][0]*0.4)
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
        
        #print(f"[DEBUG] legal mask is {legal_mask}")

        legal_indices = [i for i in range(len(legal_probs)) if legal_probs[i] > 0]

        index = np.random.choice(legal_indices)

        return index

    def update(self, *args, **kwargs):
        """Does nothing."""
        pass

class HumanBiddingPolicy(Policy):

    def __call__(self, state: RichmanObservation):
        highest_bid = state["action_mask"][0]
        return int(input(f"What is you bid? (Number between 0 and {highest_bid} inclusive): "))
    
    def update(self, *args, **kwargs):
        """Does nothing."""
        pass

class HumanGamePolicy(Policy):

    def __call__(self, state: RichmanObservation):
        return int(input(f"What is you action?: "))
    
    def update(self, *args, **kwargs):
        """Does nothing."""
        pass



def pickle_policy(
    policy: Policy,
    file_name: str,
    base_dir: str,
    subfolder: str = "saved_models/policies/",
):
    """Pickle to save a policy.

    Args:
        policy: Policy - Policy to save
        file_name: str - Name to give the pickle file
        base_dir: str - Location of RichmanRL base on this computer
        subfolder: str - default saved_models/policies/
    """
    dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(dir):
        os.mkdir(dir)

    with open(dir + file_name, "wb") as file:
        pickle.dump(policy, file)


def get_pickled_policy(
    file_name: str, base_dir: str, subfolder: str = "saved_models/policies/"
) -> Policy:
    """Returns a pickled policy.
    
    Args:
        file_name: str - Name to give the pickle file
        base_dir: str - Location of RichmanRL base on this computer
        subfolder: str - default saved_models/policies/
    """
    dir = os.path.join(base_dir, subfolder)
    with open(dir+file_name, 'rb') as file:
        policy = pickle.load(file)
    
    return policy
