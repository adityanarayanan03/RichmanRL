"""Implementations of value functions and value function approximators.."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch


class ValueFunction(ABC):
    """Represents value functions, both tabular and approximate."""

    @abstractmethod
    def __call__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        legal_mask: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Callable method representing the value of a state."""
        pass
