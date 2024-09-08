#################################################################################################
# Different Weights Growth for Weight Updates
#################################################################################################
from numbers import Number
import torch
import math
from typing import Union
from utils.experiment_constants import Focus


def linear_growth(w: torch.Tensor) -> torch.Tensor:
    """
    Defines weight updates when using a linear function.
    Derivatives: constant slope (derivative relative to linear rule always = 1).
    """
    return torch.ones_like(w)  # Ensure this returns a tensor of the same shape as w


def sigmoid_growth(w: torch.Tensor, plasticity: Focus, k: float) -> torch.Tensor:
    """
    METHOD
    Defines weight updates when using sigmoid function
    Derivative: 1/K² * (K - Wij) * Wij or 1/K² * (K - Wi:) * Wi:
    @param
        None
    @return
        derivative: sigmoid derivative of current weights
    """
    device = w.device
    derivative: torch.Tensor

    if plasticity == Focus.SYNASPSE:
        weight = torch.abs(w)/k
        derivative = (1 - torch.min(torch.ones_like(w), weight)) * weight
    elif plasticity == Focus.NEURON:
        input_dim = w.shape[1]
        norm: torch.Tensor = torch.norm(w/k, dim=1)
        scaled_norm: torch.Tensor = norm / math.sqrt(input_dim)

        derivative = (1 - torch.min(torch.ones_like(scaled_norm), scaled_norm)) * scaled_norm
    else:
        raise ValueError("Invalid focus type.")

    return derivative


def exponential_growth(w: torch.Tensor, plasticity: Focus) -> torch.Tensor:
    """
    METHOD
    Defines weight updates when using exponential function
    Derivative: Wij or  Wi:
    @param
        None
    @return
        derivative: exponential derivative of current weights
    """
    device = w.device
    derivative: torch.Tensor

    if plasticity == Focus.SYNASPSE:
        derivative = torch.abs(w)
    elif plasticity == Focus.NEURON:
        input_dim = w.shape[1]
        norm: torch.Tensor = torch.norm(w, dim=1)
        scaled_norm: torch.Tensor = norm / math.sqrt(input_dim)
        derivative = scaled_norm
    else:
        raise ValueError("Invalid focus type.")

    return derivative