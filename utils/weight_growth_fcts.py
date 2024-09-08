#################################################################################################
# Different Weights Growth for Weight Updates
#################################################################################################
from numbers import Number

import torch
from utils.experiment_constants import Focus, WeightGrowth


def linear_growth(w: torch.Tensor) -> torch.Tensor:
    """
    METHOD
    Defines weight updates when using linear funciton
    Derivatives 1
    @param
        None
    @return
        derivative: slope constant (derivative relative to linear rule always = 1)
    """
    return 1.0


def sigmoid_growth(w: torch.Tensor, plasticity: Focus, k: Number) -> torch.Tensor:
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
        derivative = torch.abs(current_weights)
    elif plasticity == Focus.NEURON:
        input_dim = w.shape[1]
        norm: torch.Tensor = torch.norm(w, dim=1)
        scaled_norm: torch.Tensor = norm / math.sqrt(input_dim)
        derivative = scaled_norm
    else:
        raise ValueError("Invalid focus type.")

    return derivative