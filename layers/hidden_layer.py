from abc import ABC
import math
from typing import Optional
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces.layer import NetworkLayer
from utils.experiment_constants import LayerNames, ParamInit


class HiddenLayer(NetworkLayer, ABC):
    """
    INTERFACE
    Defines a single hidden layer in ANN -> Every hidden layer should implement this class
    @instance attr.
        PARENT ATTR.
            name (LayerNames): name of layer
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from the layer
            device (str): device that will be used for CUDA
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
            alpha (float): lower bound for random unifrom 
            beta (float): upper bound for random uniform
            sigma (float): variance for random normal
            mu (float): mean for random normal
            init (ParamInit): fc parameter initiation type
        OWN ATTR.
            exponential_average (torch.Tensor): tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            sigmoid_k (float): constant for sigmoid wieght growth updates
    """

    #################################################################################################
    # Constructor Method
    #################################################################################################
    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 device: str = 'cpu',
                 learning_rate: float = 0.005,
                 alpha: float = 0,
                 beta: float = 1,
                 sigma: float = 1,
                 mu: float = 0,
                 init: ParamInit = ParamInit.UNIFORM,
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
            alpha: lower bound for random unifrom 
            beta: upper bound for random uniform
            sigma: variance for random normal
            mu: mean for random normal
            init: fc parameter initiation type
            lamb: lambda hyperparameter for lateral inhibition
            gamma: affects exponentialaverages updates
            eps: affects weight decay updates
            sigmoid_k : constant for sigmoid wieght growth updates
        @return
            None
        """
        super().__init__(input_dimension, 
                         output_dimension, 
                         device, 
                         learning_rate, 
                         alpha, 
                         beta, 
                         sigma, 
                         mu, 
                         init)
        self.name: LayerNames = LayerNames.HIDDEN

