import torch
import torch.nn as nn
from models.hyperparams import Inhibition
"""
Learning rules for MLP and CNN models
"""

def update_weights_FullyOrthogonal(input, output, initial_weight, eta=1):
    
    outer_prod = torch.einsum('ij,ik->ijk', output, input)

    ytw = torch.matmul(output.unsqueeze(1), initial_weight).squeeze(1)

    norm_term = torch.einsum('ij,ik->ijk', output, ytw)

    delta_weight = (outer_prod - eta * norm_term)

    return delta_weight


def update_weights_OrthogonalExclusive(x: torch.Tensor, y: torch.Tensor, weights, device, eta = 1):

    outer_prod: torch.Tensor = torch.einsum("ai, aj -> aij", y, x)

    W_norms: torch.Tensor = torch.norm(weights, dim=1, keepdim=False)

    scaling_terms: torch.Tensor = W_norms.reshape(weights.size(0),1) / (W_norms.reshape(1,weights.size(0)) + 10e-8)

    remove_diagonal: torch.Tensor = torch.ones(weights.size(0), weights.size(0)) - torch.eye(weights.size(0))
    remove_diagonal: torch.Tensor = remove_diagonal.reshape(weights.size(0), weights.size(0), 1).to(device)

    scaled_weight: torch.Tensor = scaling_terms.reshape(weights.size(0), weights.size(0), 1) * weights.reshape(1, weights.size(0), weights.size(1)).to(device)

    scaled_weight: torch.Tensor = scaled_weight * remove_diagonal

    norm_term: torch.Tensor = torch.einsum("ai, ak, ikj -> ij", y, y, scaled_weight)

    computed_rule = outer_prod - eta * norm_term
    
    return computed_rule

"""
input : tensor([batch, in_dim])
output : tensor([batch, out_dim])
weight : tensor([out_dim, in_dim])

In CNN, batch will be the number of kernel*kernel grids
"""
def update_weight_softhebb(input, preactivation, output, weight, inhibition=Inhibition.RePU):
    # input_shape = batch, in_dim
    # output_shape = batch, out_dim = preactivation_shape
    # weight_shape = out_dim, in_dim
    b, indim = input.shape
    b, outdim = output.shape
    multiplicative_factor = 1
    W = weight
    if inhibition == Inhibition.RePU:
        u = torch.relu(preactivation)
        multiplicative_factor = multiplicative_factor / (u + 1e-9)
    elif inhibition == Inhibition.Softmax:
        u = preactivation
    #deltas = multiplicative_factor * output * (input - torch.matmul(torch.relu(u), W).reshape(b, indim))
    deltas = (multiplicative_factor * output).reshape(b, outdim, 1) * (input - torch.matmul(torch.relu(u), W)).reshape(b, 1, indim)
    delta = torch.mean(deltas, dim=0)
    return delta