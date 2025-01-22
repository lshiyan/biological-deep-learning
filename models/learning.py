import torch
import torch.nn as nn
from models.hyperparams import Inhibition, WeightGrowth
import scipy
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
def update_weight_softhebb(input, preactivation, output, weight, target=None,
                           inhibition=Inhibition.RePU):
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

def softhebb_input_difference(x, a, normalized_weights):
    # Here x is assumed to have an L2 norm of 1
    # same for normalized_weights[i].
    # output has shape: batch, out_dim, in_dim
    batch_dim, in_dim = x.shape
    batch_dim, out_dim = a.shape
    in_space_diff = x.reshape(batch_dim, 1, in_dim) - a.reshape(batch_dim, out_dim, 1) * normalized_weights.reshape(
        1, out_dim, in_dim)
    return in_space_diff

def update_softhebb_w(y, normed_x, a, weights, inhibition: Inhibition, u=None, target=None,
                      supervised=False, weight_growth: WeightGrowth = WeightGrowth.Default):
    weight_norms = torch.norm(weights, dim=1, keepdim=True)
    normed_weights = weights / (weight_norms + 1e-9)
    batch_dim, out_dim = y.shape
    wn = weight_norms.unsqueeze(0)
    if weight_growth == WeightGrowth.Default:
        factor = 1 / (wn + 1e-9)
    elif weight_growth == WeightGrowth.Linear:
        factor = 1
    elif weight_growth == WeightGrowth.Sigmoidal:
        factor = wn * (1 - wn)
    elif weight_growth == WeightGrowth.Exponential:
        factor = wn
    else:
        raise NotImplementedError(f"Weight growth {weight_growth}, invalid.")
    if inhibition == Inhibition.RePU:
        indicator = (u > 0).float()
        
        factor = factor.squeeze(-1).expand(batch_dim, out_dim)  * indicator.reshape(batch_dim, out_dim) / (u.reshape(batch_dim, out_dim) + 1e-9)
        factor = factor.mean(dim=0, keepdim=True).unsqueeze(-1) 
    if supervised:
        y_part = (target - y).reshape(batch_dim, out_dim)
    else:
        y_part = y.reshape(batch_dim, out_dim)
    
    batch_dim, in_dim = normed_x.shape
    batch_dim, out_dim = a.shape

    ### Anti hebbian test: 
    max_values, indices = torch.max(y_part, dim=1, keepdim=True)
    # Create a mask where the maximum values are located
    mask = torch.zeros_like(y_part, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    # Set the non-maximum values to negative
    anti_hebbian_output = torch.where(mask, y_part, -y_part)
        
    # Innefficient. We are trying to calculate it in a more memory efficient way.
    # delta_w = factor * y_part * x.reshape(batch_dim, 1, in_dim) - a.reshape(batch_dim, out_dim, 1) * normalized_weights.reshape(1, out_dim, in_dim)
    
    ya = torch.mean(anti_hebbian_output * a, dim=0).reshape(out_dim, 1)
    yx = (1/batch_dim) * torch.matmul(anti_hebbian_output.T, normed_x)
    delta_w = factor * (yx - ya * normed_weights)
    delta_w = torch.mean(delta_w, dim=0) # average the delta weights over the batch dim

    return delta_w

def update_softhebb_b(y, logprior, target=None, supervised=False):
    priors = torch.softmax(logprior, dim=0).unsqueeze(0)
    if supervised:
        delta_b = target - priors
    else:
        delta_b = y - priors
    delta_b = torch.mean(delta_b, dim=0)  # mean over batch dim

    return delta_b


def update_softhebb_lamb(y, a, inhibition: Inhibition, lamb=None, in_dim=None, target=None, supervised=False):
    if inhibition == Inhibition.Softmax:
        v = a
    elif inhibition == Inhibition.RePU:
        u = torch.relu(a)
        v = (u > 0).float() * torch.log(u + 1e-7)
    else:
        raise NotImplementedError(f"{inhibition} not implemented type of inhibition in update λ.")

    if supervised:
        delta_l = torch.sum((target - y) * v, dim=1)
    else:
        if inhibition == Inhibition.Softmax:
            k = scipy.special.iv(in_dim/2, lamb)/scipy.special.iv(in_dim/2 -1, lamb)
        elif inhibition == Inhibition.RePU:
            k = 0.5 * (scipy.special.psi(0.5 * (lamb + 1)) - scipy.special.psi(0.5 * (lamb + in_dim)))
        else:
            raise NotImplementedError(f"{inhibition} not implemented type of inhibition in update λ.")
        delta_l = torch.sum(y * v, dim=1) - k
    delta_l = torch.mean(delta_l)  # mean over batch dim
    return delta_l


