"""Implementation of MLPs for function approximation."""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """MLP for function/policy approximation."""

    def __init__(self, input_dims: int, output_dims: int):
        """Constructor.

        Args:
            input_dims: int - Input dimension
            output_dims: int - Output dimension
        """
        super(MLP, self).__init__()  # noqa: UP008

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(input_dims, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: np.ndarray, legal_mask: np.ndarray, return_prob=True
    ) -> torch.Tensor:
        """Forward pass through NN.

        If self.output_dims > 1 then will softmax the output, unless
        return_prob is False, in which case will return a torch.multinomial.

        Args:
            x: np.ndarray - numpy array of input
            legal_mask: np.ndarray - Mask for legal moves
            return_prob: bool - For policy approximation.
        """
        x = torch.from_numpy(x).float()
        legal_mask = torch.from_numpy(legal_mask).float()

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        print(f"legal mask as a tensor is {legal_mask}")
        out = out * legal_mask
        print(f"After pairwise product, output is {out}")

        if self.output_dims > 1:
            out = self.softmax(out)
        print(f"After softmax, output is {out}")

        if not return_prob:
            return torch.multinomial(out, 1).item()

        return out
