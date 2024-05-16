import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numpy import outer
import warnings

warnings.filterwarnings("ignore")

"""
Class defining the functionality of the classification layer
"""
class ClassifierLayer(NetworkLayer):
    """
    Constructor method NetworkLayer
    @param
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        lamb (float) = lambda hyperparameter for latteral inhibition
        learning_rate (float) = how fast model learns at each iteration
        gamma (float) = decay factor -> factor to decay learning rate
        eps (float) = to avoid division by 0
    @attr.
    PARENT ATTR.
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        lamb (float) = lambda hyperparameter for latteral inhibition
        alpha (float) = how fast model learns at each iteration
        fc (fct) = function to apply linear transformation to incoming data
        scheduler (layers.Scheduler) = scheduler for current layer
        eps (float) = to avoid division by 0
        exponential_average (torch.Tensor) = 0 tensor to keep track of exponential averages
        gamma (float) = decay factor -> factor to decay learning rate
        id_tensor (torch.Tensor) = id tensor of layer
    OWN ATTR.
    """
    def __init__(self, input_dimension, output_dimension, classifier, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, lamb, heb_lr, gamma, eps)     
    
    """
    Defines the way the weights will be updated at each iteration of the training
    @param
        input (torch.Tensor): The input tensor to the layer before any transformation.
        output (torch.Tensor): The output tensor of the layer before applying softmax.

    The method computes the outer product of the softmax probabilities of the outputs and the inputs. 
    This product is scaled by the learning rate and used to adjust the weights. 
    The weights are then normalized to ensure stability.
    """
    def update_weights(self, input, output):

        # Detach and squeeze tensors to remove any dependencies and reduce dimensions if possible.
        u = output.clone().detach().squeeze() # Output tensor after layer but before activation
        x = input.clone().detach().squeeze() # Input tensor to layer
        y = torch.softmax(u, dim=0) # Apply softmax to output tensor to get probabilities
        A = None
        # FIXME: clamped_output whats the use? do we need it? and if we do have to write it into layer.py file & change hebbian_layer.py
        """
        if clamped_output != None:
            outer_prod = torch.outer(clamped_output-y,x)
            u_times_y  =torch.mul(u,y)
            A = outer_prod  -self.fc.weight * (u_times_y.unsqueeze(1))
        else:
        """

        # Compute the outer product of the softmax output and input.
        A = torch.outer(y,x) # Hebbian learning rule component

        # Adjust weights by learning rate and add contribution from Hebbian update.
        A = self.fc.weight + self.alpha * A

        # Normalize weights by the maximum value in each row to stabilize the learning.
        weight_maxes = torch.max(A, dim=1).values
        self.fc.weight = nn.Parameter(A/weight_maxes.unsqueeze(1), requires_grad=False)

        # Zero out the first column of weights -> this is to prevent the first weight from learning everything
        self.fc.weight[:, 0] = 0
        

    """
    Defines the way the biases will be updated at each iteration of the training
    It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.

    @param
        output (torch.Tensor): The output tensor of the layer before applying softmax.

    The method applies an exponential decay to the biases, which is modulated by the output probabilities,
    and scales the update by the learning rate. 
    The biases are normalized after the update.
    """
    def update_bias(self, output):
        y = output.clone().detach().squeeze() # Output tensor probabilities after softmax
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y)-1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability.
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
    
    """
    Feed forward
    @param
        train (bool) = model in trainning or not
    """
    def forward(self, x):
        input_copy = x.clone()
        x = self.fc(x)
        self.update_weights(input_copy, x)
        #self.updateBias(x, train=train)
        x = self.softmax(x)
        return x
    
    """
    Counts the number of active feature selectors (above a certain cutoff beta).
    @param
    beta (float) = cutoff value determining which neuron is active and which is not
    """
    def active_classifier_weights(self, beta):
        weights = self.fc.weight
        active = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)
