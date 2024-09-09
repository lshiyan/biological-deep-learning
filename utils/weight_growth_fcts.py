#################################################################################################
# Different Weights Growth for Weight Updates
#################################################################################################
from numbers import Number
import torch
import math
from typing import Union
from utils.experiment_constants import Focus

def neuron_norm(w, k):
    out_dim, in_dim = w.shape
    norm = torch.norm(w / k, dim=1, keepdim=True) / math.sqrt(in_dim)
    return norm

# Define growth functions outside the class for better structure
def linear_growth(w: torch.Tensor) -> torch.Tensor:
    """Defines weight updates using a linear function."""
    return torch.ones_like(w)  # Returns a tensor of ones with the same shape as w

def sigmoid_growth(w: torch.Tensor, plasticity: 'Focus', k: float) -> torch.Tensor:
    """Defines weight updates using a sigmoid function."""
    if plasticity == Focus.SYNAPSE:
        weight = torch.abs(w) / k
        derivative = (1 - torch.min(torch.ones_like(w), weight)) * weight
    elif plasticity == Focus.NEURON:
        if w.ndim == 1:
            # print(f"Warning: Expected at least 2 dimensions for NEURON focus, got shape {w.shape}")
            derivative = (1 - min(torch.ones_like(w), torch.abs(w) / k)) * torch.abs(w) / k
        elif w.ndim == 2:
            scaled_norm = neuron_norm(w, k)
            derivative = (1 - torch.min(torch.ones_like(scaled_norm), scaled_norm)) * scaled_norm
        else:
            raise ValueError("Unexpected tensor dimensions for NEURON focus.")
    else:
        raise ValueError("Invalid focus type.")
    return derivative

def exponential_growth(w: torch.Tensor, plasticity: 'Focus') -> torch.Tensor:
    """Defines weight updates using an exponential function."""
    if plasticity == Focus.SYNAPSE:
        derivative = torch.abs(w)
    elif plasticity == Focus.NEURON:
        if w.ndim == 1:
            print(f"Warning: Expected at least 2 dimensions for NEURON focus, got shape {w.shape}")
            derivative = torch.abs(w)
        elif w.ndim == 2:
            derivative = neuron_norm(w, 1)
        else:
            raise ValueError("Unexpected tensor dimensions for NEURON focus.")
    else:
        raise ValueError("Invalid focus type.")
    return derivative