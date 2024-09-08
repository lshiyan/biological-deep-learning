#################################################################################################
# Different Weights Growth for Weight Updates
#################################################################################################
from numbers import Number
import torch
import math
from typing import Union
from utils.experiment_constants import Focus

# Define growth functions outside the class for better structure
def linear_growth(w: torch.Tensor) -> torch.Tensor:
    """Defines weight updates using a linear function."""
    return torch.ones_like(w)  # Returns a tensor of ones with the same shape as w



def sigmoid_growth(w: torch.Tensor, plasticity: 'Focus', k: float) -> torch.Tensor:
    """Defines weight updates using a sigmoid function."""
    if plasticity == Focus.SYNASPSE:
        weight = torch.abs(w) / k
        derivative = (1 - torch.min(torch.ones_like(w), weight)) * weight
    elif plasticity == Focus.NEURON:
        if w.ndim == 1:
            print(f"Warning: Expected at least 2 dimensions for NEURON focus, got shape {w.shape}")
            norm = torch.norm(w / k)
            derivative = (1 - min(1.0, norm)) * norm
            return torch.full_like(w, derivative.item())  # Fill with repeated value
        elif w.ndim == 2:
            input_dim = w.shape[1]
            norm = torch.norm(w / k, dim=1, keepdim=True)
            scaled_norm = norm / math.sqrt(input_dim)
            derivative = (1 - torch.min(torch.ones_like(scaled_norm), scaled_norm)) * scaled_norm
            derivative = derivative.repeat(1, w.size(1))  # Adjust shape to match w
        else:
            raise ValueError("Unexpected tensor dimensions for NEURON focus.")
    else:
        raise ValueError("Invalid focus type.")
    return derivative



def exponential_growth(w: torch.Tensor, plasticity: 'Focus') -> torch.Tensor:
    """Defines weight updates using an exponential function."""
    if plasticity == Focus.SYNASPSE:
        derivative = torch.abs(w)
    elif plasticity == Focus.NEURON:
        if w.ndim == 1:
            print(f"Warning: Expected at least 2 dimensions for NEURON focus, got shape {w.shape}")
            norm = torch.norm(w)
            derivative = torch.full_like(w, norm)  # Use the same norm value for all elements
        elif w.ndim == 2:
            input_dim = w.shape[1]
            norm = torch.norm(w, dim=1, keepdim=True)
            scaled_norm = norm / math.sqrt(input_dim)
            derivative = scaled_norm.repeat(1, w.size(1))  # Adjust shape to match w
        else:
            raise ValueError("Unexpected tensor dimensions for NEURON focus.")
    else:
        raise ValueError("Invalid focus type.")
    return derivative