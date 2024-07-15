from abc import ABC
import logging
from typing import Optional, Tuple
import warnings
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
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): device that will be used for CUDA
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            id_tensor (torch.Tensor): id tensor of layer
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
                 device: str,
                 learning_rate: float = 0.005,
                 alpha: float = 0,
                 beta: float = 1,
                 sigma: float = 1,
                 mu: float = 0,
                 init: ParamInit = ParamInit.UNIFORM,
                 lamb: float = 1,
                 gamma: float = 0.99, 
                 eps: float = 0.01,
                 sigmoid_k: float = 1
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
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
        self.gamma: float = gamma
        self.lamb: float = lamb
        self.eps: float = eps
        self.sigmoid_k: float = sigmoid_k
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension).to(self.device)
        self.id_tensor: torch.Tensor = self.create_id_tensors().to(self.device)



    #################################################################################################
    # Different Inhibition Methods
    #################################################################################################
    def _relu_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates ReLU lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # Get ReLU activation function
        relu: nn.ReLU = nn.ReLU()
        
        # Compute ReLU and lateral inhibition
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        input_copy = relu(input_copy)
        max_ele: float = torch.max(input_copy).item()
        input_copy = torch.pow(input_copy, self.lamb).to(self.device)
        output: torch.Tensor =  (input_copy / abs(max_ele) ** self.lamb).to(self.device)
        
        return output
    
    
    def _softmax_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates softmax lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        output: torch.Tensor = F.softmax(input_copy, dim=-1).to(self.device)
        return output
    
    
    def _exp_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates exponential (Modern Hopfield) lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        max_ele: float = torch.max(input_copy).item()
        output: torch.Tensor = F.softmax((input_copy - max_ele) * self.lamb, dim=-1).to(self.device)
        return output
    
        
    def _wta_inhibition(self, input:torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        METHOD
        Calculates k-winners-takes-all lateral inhibition
        @param
            input: input to layer
            threshold: amount to be counted as activated
        @return
            output: activation after lateral inhibition
        """
        # NOTE: this function does not work as of yet
        output: torch.Tensor = torch.where(input>=threshold, 1.0, 0.0).to(self.device)

        return output

    
    def _gaussian_inhibition(self, input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        METHOD
        Calculates gaussian lateral inhibition
        @param
            input: input to layer
            sigma: 
        @return
            output: activation after lateral inhibition
        """
        # NOTE: this does not work as of yet
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        size: int = int(2 * sigma + 1)
        kernel: torch.Tensor = torch.tensor([torch.exp(torch.Tensor(-(i - size // 2) ** 2 / (2 * sigma ** 2))) for i in range(size)])
        kernel = kernel / torch.sum(kernel)

        output: torch.Tensor = F.conv1d(input_copy.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=size//2).squeeze(0).squeeze(0).to(self.device)
        return output

    
    def _norm_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates devisive normalization lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # Get ReLU activation function
        relu: nn.ReLU = nn.ReLU()
        
        # Compute ReLU and lateral inhibition
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        input_copy = relu(input_copy)
        sum: torch.Tensor = input_copy.sum()
        output: torch.Tensor =  (input_copy / sum).to(self.device)
        
        return output
    
    
    #################################################################################################
    # Different Weight Updates Methods
    #################################################################################################
    def _hebbian_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Computes Hebbian Leanring Rule.
        @param
            input: the input of the layer
            output: the output of the layer
        @return
            
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Calculate Hebbian Learning Rule
        computed_rule: torch.Tensor = outer_prod.to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule

    
    def _sanger_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Computes Sanger's Rule
        @param
            input: the input of the layer
            output: the output of the layer
        @return
            computed_rule: this is the delta_weight value
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Retrieve initial weights (transposed) 
        initial_weight: torch.Tensor = torch.transpose(self.fc.weight.clone().detach().to(self.device), 0, 1)

        # Calculate Sanger's Rule
        A: torch.Tensor = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y).to(self.device)
        A = A * (y.unsqueeze(1))
        computed_rule: torch.Tensor = (outer_prod - A).to(self.device).to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule


    def _fully_orthogonal_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Update weights using Fully Orthogonal Rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Retrieve initial weights
        initial_weight: torch.Tensor = self.fc.weight.clone().detach().to(self.device)

        # Calculate Fully Orthogonal Rule
        ytw = torch.matmul(y.unsqueeze(0), initial_weight).to(self.device)
        norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0)).to(self.device)

        # Compute change in weights
        computed_rule: torch.Tensor = (outer_prod - norm_term).to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule



    #################################################################################################
    # Different Weights Growth for Wegiht Updates
    #################################################################################################
    def _linear_function(self) -> torch.Tensor:
        """
        METHOD
        Defines weight updates when using linear funciton
        @param
            None
        @return
            derivative: slope constant (derivative relative to linear rule always = 1)
        """
        return torch.ones(self.fc.weight.shape).to(self.device)
    
    
    def _sigmoid_function(self) -> torch.Tensor:
        """
        METHOD
        Defines weight updates when using sigmoid funciton
        @param
            None
        @return
            derivative: sigmoid derivative of current weights
        """
        current_weights: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
        sigmoid_k_tensor: torch.Tensor = torch.full(self.fc.weight.shape, self.sigmoid_k).to(self.device)
        derivative: torch.Tensor = (sigmoid_k_tensor - current_weights).to(self.device)
        derivative = current_weights * derivative
        derivative = (1 / self.sigmoid_k) * derivative
        return derivative
        

        
    #################################################################################################
    # Different Weights Growth for Wegiht Updates
    #################################################################################################
    def _linear_weight_decay(self) -> torch.Tensor:
        """
        METHOD
        Decays the overused weights and increases the underused weights using tanh functions.
        @param
            None
        @return
            None
        """
        tanh: nn.Tanh = nn.Tanh()

        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()

        # Gets ratio vs mean
        norm_exp_avg: torch.Tensor = (self.exponential_average / average).to(self.device)
        
        x: torch.Tensor = (-self.eps * (norm_exp_avg - 1)).to(self.device)
        # x: torch.Tensor = self.eps * (norm_exp_avg - 1)
        sech_x: torch.Tensor = (1 / torch.cosh(x)).to(self.device)

        # Calculate the growth factors
        growth_factor_positive: torch.Tensor = (self.eps * tanh(x) + 1).to(self.device)
        growth_factor_negative: torch.Tensor = (torch.reciprocal(growth_factor_positive)).to(self.device)
        growth_factor_positive = growth_factor_positive.unsqueeze(1)
        growth_factor_negative = growth_factor_negative.unsqueeze(1)
        
        # cst_term: float = -self.eps * (1 - self.gamma) / average
        # last_term: torch.Tensor = torch.reciprocal(torch.sinh(x) * torch.cosh(x))
        # extra_term: torch.Tensor = (1 - self.eps * tanh(x)) ** 2
        
        # growth_factor_positive: torch.Tensor = self.fc.weight * cst_term * last_term.unsqueeze(1)
        # growth_factor_negative: torch.Tensor = self.fc.weight * cst_term * last_term.unsqueeze(1) * extra_term.unsqueeze(1)
        
        # Combined growth_factor
        growth_factor: torch.Tensor = torch.where(self.fc.weight > 0, growth_factor_positive, growth_factor_negative).to(self.device)
        
        return growth_factor
    

    def _sigmoid_weight_decay(self) -> torch.Tensor:
        # NOTE: currently does not work
        """
        METHOD
        Decays the overused weights and increases the underused weights using tanh functions.
        @param
            None
        @return
            None
        """
        tanh: nn.Tanh = nn.Tanh()
        sigmoid: nn.Sigmoid = nn.Sigmoid()

        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()

        # Gets ratio vs mean
        norm_exp_avg: torch.Tensor = (self.exponential_average / average).to(self.device)
        
        x: torch.Tensor = (-self.eps * (norm_exp_avg - 1)).to(self.device)

        # calculate the growth factors
        growth_factor_positive: torch.Tensor = sigmoid(self.eps * tanh(x) + 1).to(self.device)
        growth_factor_negative: torch.Tensor = torch.reciprocal(growth_factor_positive).to(self.device)
        growth_factor_positive = growth_factor_positive.unsqueeze(1)
        growth_factor_negative = growth_factor_negative.unsqueeze(1)
        
        growth_factor = torch.where(self.fc.weight > 0, growth_factor_positive, growth_factor_negative).to(self.device)
        
        return growth_factor
    
    
    def _simple_linear_weight_decay(self) -> torch.Tensor:
        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()
        
        # Compute simple weight decay
        decay_weight: torch.Tensor = (self.eps * (self.exponential_average / average - 1)).to(self.device)
        decay_weight = decay_weight.unsqueeze(1)
        
        return decay_weight
    
    
    def _simple_sigmoid_weight_decay(self) -> torch.Tensor:
        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()
        
        # Compute simple weight decay
        decay_weight: torch.Tensor = (self.eps * (self.exponential_average / average - 1)).to(self.device)
        decay_weight = decay_weight.unsqueeze(1)
        decay_weight = self._sigmoid_function() * decay_weight
        
        return decay_weight
                


    #################################################################################################
    # Different Weights Growth for Wegiht Updates
    #################################################################################################
    def _hebbian_bias_update(self, output:torch.Tensor) -> None:
        """
        METHOD
        Defines the way the biases will be updated at each iteration of the training
        It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.
        The method applies an exponential decay to the biases, which is modulated by the output probabilities,
        and scales the update by the learning rate. 
        The biases are normalized after the update.
        @param
            output: The output tensor of the layer.
        @return
            None
        """
        y: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        exponential_bias = (torch.exp(-1 * self.fc.bias)).to(self.device)

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = (torch.mul(exponential_bias, y) - 1).to(self.device)
        A = self.fc.bias + self.lr * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: torch.Tensor = (torch.max(A, dim=0).values).to(self.device)
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
        
        
    def _simple_bias_update(self, output: torch.Tensor) -> None:
        """
        METHOD
        Defines a simple bias update rule
        @param
            output: output of the current layer
        @return
            None
        """
        output_copy: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        
        avg_output: float = torch.mean(output_copy).item()
        bias_update: torch.Tensor = (self.eps * (output_copy - avg_output)).to(self.device)
        self.fc.bias = nn.Parameter(torch.sub(self.fc.bias, bias_update), requires_grad=False)
    
    
    
    #################################################################################################
    # Activations and weight/bias updates that will be called for train/eval forward
    #################################################################################################
    def inhibition(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method has yet to be implemented.")


    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError("This method has yet to be implemented.")
    

    def update_bias(self, output: torch.Tensor) -> None:
        raise NotImplementedError("This method has yet to be implemented.")
    

    def weight_decay(self) -> None:
        raise NotImplementedError("This method has yet to be implemented.")



    #################################################################################################
    # Training/Evaluation during forward pass
    #################################################################################################
    def _train_forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("This method has yet to be implemented.")
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method has yet to be implemented.")