import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.output_layer import OutputLayer
from utils.experiment_constants import ActivationMethods, BiasUpdate, Focus, LearningRules, ParamInit, WeightDecay, WeightGrowth


class ClassificationLayer(OutputLayer):
    """
    CLASS
    Defines the functionality of the base classification layer
    @instance attr.
        NetworkLayer ATTR.
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
                 sigmoid_k: float = 1,
                 init: ParamInit = ParamInit.UNIFORM,
                 learning_rule: LearningRules = LearningRules.HEBBIAN_LEARNING_RULE,
                 weight_growth: WeightGrowth = WeightGrowth.LINEAR,
                 bias_update: BiasUpdate = BiasUpdate.NO_BIAS,
                 focus: Focus = Focus.SYNASPSE,
                 activation: ActivationMethods = ActivationMethods.BASIC
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
            include_first: wether or not to include first neuro in classification
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device, learning_rate, alpha, beta, sigma, mu, init)
        self.learning_rule: LearningRules = learning_rule
        self.weight_growth: WeightGrowth = weight_growth
        self.bias_update: BiasUpdate = bias_update
        self.focus: Focus = focus
        self.activation_method: ActivationMethods = activation
        
        self.sigmoid_k: float = sigmoid_k
        

    #################################################################################################
    # Activations and weight/bias updates that will be called for train/eval forward
    #################################################################################################
    def activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates activation of the layer.
        @param
            input: the inputs into the layer
        @return
            outputs of layer
        """ 
        if self.activation_method == ActivationMethods.BASIC:
            return self._basic_activation(input)
        elif self.activation_method == ActivationMethods.NORMALIZED:
            return self._normalized_activation(input)
        else:
            raise ValueError("Invalid activation method.")
        
        
    def probability(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates lateral inhibition using defined rule.
        @param
            input: the inputs into the layer
        @return
            outputs of inhibiton
        """ 
        softmax: nn.Softmax = nn.Softmax(dim=-1)
        output: torch.Tensor = softmax(input).to(self.device)
        
        return output
    
    
    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using defined rule.
        delta W = lr * rule * derivative
        W = W + delta W
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        calculated_rule: torch.Tensor
        function_derivative: torch.Tensor
        
        if self.learning_rule == LearningRules.HEBBIAN_LEARNING_RULE:
            calculated_rule = self._hebbian_rule(input, output, clamped_output)
        elif self.learning_rule == LearningRules.CONTROLLED_LEARNING_RULE:
            calculated_rule = self._controlled_hebbian_rule(input, output, clamped_output)
        else:
            raise NameError("Unknown learning rule.")
        
        if self.weight_growth == WeightGrowth.LINEAR:
            function_derivative = self._linear_function()
        elif self.weight_growth == WeightGrowth.SIGMOID:
            function_derivative = self._sigmoid_function()
        elif self.weight_growth == WeightGrowth.EXPONENTIAL:
            function_derivative = self._exponential_function()
        else:
            raise NameError("Unknown weight growth rule.")
            
        # Weight Update
        delta_weight: torch.Tensor = (self.lr * calculated_rule * function_derivative).to(self.device)
        updated_weight: torch.Tensor = torch.add(self.fc.weight, delta_weight)
        # self.fc.weight = nn.Parameter(updated_weight, requires_grad=False)
        
        normalized_weight: torch.Tensor = self.normalize(updated_weight)
        self.fc.weight = nn.Parameter(normalized_weight, requires_grad=False)
        

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Update bias using pre-defined rule
        @param
            output: The output tensor of the layer.
        @return
            None
        """
        if self.bias_update == BiasUpdate.NO_BIAS:
            return
        else:
            raise ValueError("Update Bias type invalid.")
        
    
    
    #################################################################################################
    # Training and Evaluation Methods
    #################################################################################################
    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: *NOT USED*
        @return
            output: returns the data after passing it through the layer
        """
        # Calculate activation -> Calculate inhibition -> Update weights -> Update bias -> Return output
        activations: torch.Tensor = self.activation(input)
        output: torch.Tensor = self.probability(activations)
        self.update_weights(input, activations, clamped_output)
        self.update_bias(output)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            raise ValueError("Weights of the fully connected layer have become NaN.")
        
        return output
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Define how an input data flows through the network when testing
        @param
            input: input data into the layer
        @return
            output: returns the data after passing it throw the layer
        """
        # Calculate activation -> calculate inhibition -> Return output
        activations: torch.Tensor = self.activation(input)
        output: torch.Tensor = self.probability(activations)
        
        return output
    
    
    #################################################################################################
    # Different Activation Methods
    #################################################################################################
    def _basic_activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates the activations through fully connected linear layer
        @param
            input: input to layer
        @return
            output: activation after passing through layer
        """
        return self.fc(input)
    
    
    def _normalized_activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates the normalized activations through fully connected linear layer
        @param
            input: input to layer
        @return
            output: activation after passing through layer
        """
        weight: torch.Tensor = self.fc.weight.clone().detach().float().to(self.device)
        normalized_weight: torch.Tensor = self.normalize(weight).to(self.device)
        
        return F.linear(input, normalized_weight, bias=self.fc.bias)
    
    
    #################################################################################################
    # Different Weight Updates Methods
    #################################################################################################
    def _hebbian_rule(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines the way the weights will be updated at each iteration of the training.
        Rule = delta Sij = lr * t * xj
        @param
            input: The input tensor to the layer before any transformation.
            output: The output tensor of the layer before applying softmax.
            clamped_output: one-hot encode of true labels
        @return
            computed_rule: computed hebbian rule
        """
        # Detach and squeeze tensors to remove any dependencies and reduce dimensions if possible.
        y: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        x: torch.Tensor = input.clone().detach().squeeze().to(self.device)

        
        computed_rule: torch.Tensor = torch.outer(clamped_output, x).to(self.device)

        return computed_rule

    
    def _controlled_hebbian_rule(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines the way the weights will be updated at each iteration of the training.
        Rule = delta Sij = lr * (t - yi) xj - Wij * (ui * yi)
        @param
            input: The input tensor to the layer before any transformation.
            output: The output tensor of the layer before applying softmax.
            clamped_output: one-hot encode of true labels
        @return
            computed_rule: computed controlled hebbian rule
        """
        # Detach and squeeze tensors to remove any dependencies and reduce dimensions if possible.
        u: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        x: torch.Tensor = input.clone().detach().squeeze().to(self.device)
        y: torch.Tensor = torch.softmax(u, dim=-1).to(self.device)

        outer_prod: torch.Tensor = torch.outer(clamped_output - y, x).to(self.device)
        computed_rule: torch.Tensor = (outer_prod - self.fc.weight * ((u * y).unsqueeze(-1))).to(self.device)
        
        return computed_rule



    #################################################################################################
    # Different Weights Growth
    #################################################################################################
    def _linear_function(self) -> torch.Tensor:
        """
        METHOD
        Defines weight updates when using linear funciton
        Derivatives 1
        @param
            None
        @return
            derivative: slope constant (derivative relative to linear rule always = 1)
        """
        return torch.ones(self.fc.weight.shape).to(self.device)
    
    
    def _sigmoid_function(self) -> torch.Tensor:
        """
        METHOD
        Defines weight updates when using sigmoid function
        Derivative: 1/K * (K - Wij) * Wij or 1/K * (K - ||Wi:||) * ||Wi:||
        @param
            None
        @return
            derivative: sigmoid derivative of current weights
        """
        current_weights: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
        derivative: torch.Tensor
        
        if self.focus == Focus.SYNASPSE:
            derivative = (1 / self.sigmoid_k) * (self.sigmoid_k - current_weights) * current_weights
        elif self.focus == Focus.NEURON:
            norm: torch.Tensor = self.get_norm(self.fc.weight)
            scaled_norm: torch.Tensor = norm / math.sqrt(self.output_dimension)

            derivative = (1 / self.sigmoid_k) * (self.sigmoid_k - torch.min(torch.ones_like(scaled_norm), scaled_norm)) * scaled_norm
        else:
            raise ValueError("Invalid focus type.")
        
        return derivative
    
    
    def _exponential_function(self) -> torch.Tensor:
        """
        METHOD
        Defines weight updates when using exponential function
        Derivative: Wij or  Wi:
        @param
            None
        @return
            derivative: exponential derivative of current weights
        """
        current_weights: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
        derivative: torch.Tensor
        
        if self.focus == Focus.SYNASPSE:
            derivative = torch.abs(current_weights)
        elif self.focus == Focus.NEURON:
            norm: torch.Tensor = self.get_norm(self.fc.weight)
            derivative = norm
        else:
            raise ValueError("Invalid focus type.")
        
        return derivative
                


    #################################################################################################
    # Different bias updates
    #################################################################################################
    def _hebbian_bias_update(self, output:torch.Tensor) -> None:
        """
        METHOD
        Define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        y: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        exponential_bias: torch.Tensor = torch.exp(-1 * self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = torch.mul(exponential_bias, y)-1
        A = (1 - self.lr) * self.fc.bias + self.lr * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: torch.Tensor = (torch.max(A, dim=0).values).to(self.device)
        self.fc.bias = nn.Parameter(A / bias_maxes.item(), requires_grad=False)
        
        
    def _simple_bias_update(self, output: torch.Tensor) -> None:
        """
        METHOD
        Defines a simple bias update rule
        Update: b - eps * (yi - y)
        @param
            output: output of the current layer
        @return
            None
        """
        output_copy: torch.Tensor = output.clone().detach().squeeze().to(self.device)
        
        avg_output: float = torch.mean(output_copy).item()
        bias_update: torch.Tensor = (self.eps * (output_copy - avg_output)).to(self.device)
        self.fc.bias = nn.Parameter(torch.sub(self.fc.bias, bias_update), requires_grad=False)