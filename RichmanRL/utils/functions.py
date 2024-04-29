"""Implementations of value functions and value function approximators.."""

from abc import ABC, abstractmethod
from RichmanRL.envs.typing_utils import RichmanObservation


class ValueFunction(ABC):
    """Represents value functions, both tabular and approximate."""

    @abstractmethod
    def __call__(
        self,
        state: RichmanObservation,
    ) -> float:
        """Callable method representing the value of a state."""
        pass

    @abstractmethod
    def update(self, state: RichmanObservation, G: float):
        """Update the value of a state, addressed by a feature vector.

        Args:
            state: RichmanObservation - feature vector embedding
                of a state.
            G : float - return to update towrads.
        """
        pass

class ConstantBaseline(ValueFunction):
    """Represents a constant baseline."""
    def __init__(self, baseline = 0):
        """Just set the baseline constant."""
        self.baseline = baseline

    def __call__(self, state:RichmanObservation) -> float:
        """Return the constant baseline."""
        return self.baseline
    
    def update(self, features, G):
        """Nothing happens on update."""
        pass