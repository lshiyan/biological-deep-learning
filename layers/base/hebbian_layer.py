from typing import Optional
import warnings
import torch
import torch.nn as nn
from layers.hidden_layer import HiddenLayer
from utils.experiment_constants import *


class HebbianLayer(HiddenLayer):
    """
    CLASS
    Defines the functionality of the base hebbian layer
    @instance attr.
        NetworkLayer ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): device that will be used for CUDA
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        HiddenLayer ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            id_tensor (torch.Tensor): id tensor of layer
        OWN ATTR.
            inhibition_rule (LateralInhibitions): which inhibition to be used
            learning_rule (LearningRules): which learning rule to use
            weight_growth (WeightGrowth): which function type should the weight updates follow
    """
    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 device: str, 
                 lamb: float = 1, 
                 learning_rate: float = 0.005,
                 alpha: float = 0,
                 beta: float = 1,
                 sigma: float = 1,
                 mu: float = 0,
                 init: ParamInit = ParamInit.UNIFORM, 
                 gamma: float = 0.99, 
                 eps: float = 0.01,
                 sigmoid_k: float = 1,
                 inhibition_rule: LateralInhibitions = LateralInhibitions.RELU_INHIBITION, 
                 learning_rule: LearningRules = LearningRules.SANGER_LEARNING_RULE,
                 weight_growth: WeightGrowth = WeightGrowth.LINEAR
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            lamb: lambda hyperparameter for lateral inhibition
            learning_rate: how fast model learns at each iteration
            gamma: affects exponentialaverages updates
            eps: affects weight decay updates
            inhibition_rule (LateralInhibitions): which inhibition to be used
            learning_rule (LearningRules): which learning rule to use
            weight_growth (WeightGrowth): which function type should the weight updates follow
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
                         init, 
                         lamb, 
                         gamma, 
                         eps, 
                         sigmoid_k)
        self.inhibition_rule: LateralInhibitions = inhibition_rule
        self.learning_rule: LearningRules = learning_rule
        self.weight_growth: WeightGrowth = weight_growth

    
    def inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates lateral inhibition using defined rule.
        @param
            input: the inputs into the layer
        @return
            inputs after inhibition
        """ 
        if self.inhibition_rule == LateralInhibitions.RELU_INHIBITION:
            return self._relu_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.SOFTMAX_INHIBITION:
            return self._softmax_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.HOPFIELD_INHIBITION:
            return self._exp_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.WTA_INHIBITION:
            return self._wta_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.GAUSSIAN_INHIBITION:
            return self._gaussian_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.NORM_INHIBITION:
            return self._norm_inhibition(input)
        else:
            raise NameError("Unknown inhibition rule.")
    
    
    def update_weights(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using defined rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        calculated_rule: torch.Tensor
        function_derivative: torch.Tensor
        
        if self.learning_rule == LearningRules.HEBBIAN_LEARNING_RULE:
            calculated_rule = self._hebbian_rule(input, output)
        elif self.learning_rule == LearningRules.SANGER_LEARNING_RULE:
            calculated_rule = self._sanger_rule(input, output)
        elif self.learning_rule == LearningRules.FULLY_ORTHOGONAL_LEARNING_RULE:
            calculated_rule = self._fully_orthogonal_rule(input, output)
        
        if self.weight_growth == WeightGrowth.LINEAR:
            function_derivative = self._linear_function()
        elif self.weight_growth == WeightGrowth.SIGMOID:
            function_derivative = self._sigmoid_function()
            
        # Weight Update
        lr_tensor: torch.Tensor = torch.full(self.fc.weight.shape, self.lr)
        delta_weight: torch.Tensor = lr_tensor * calculated_rule * function_derivative
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        

    def update_bias(self, output: torch.Tensor) -> None:
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
        y: torch.Tensor = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1 * self.fc.bias)

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.lr * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: torch.Tensor = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
        

    def weight_decay(self) -> None:
        """
        METHOD
        Decays the overused weights and increases the underused weights using tanh functions.
        @param
            None
        @return
            None
        """
        growth_factor: torch.Tensor
        if self.weight_growth == WeightGrowth.LINEAR:
            growth_factor = self._linear_weight_decay()
        elif self.weight_growth == WeightGrowth.SIGMOID:
            pos_weights, neg_weights = self._sigmoid_weight_decay()
        else:
            raise NameError(f"Invalid weight growth {self.weight_growth}.")
        
        # weight_change: torch.Tensor = self.fc.weight * growth_factor
        # self.fc.weight = nn.Parameter(torch.add(self.fc.weight, weight_change), requires_grad=False)
        self.fc.weight = nn.Parameter(self.fc.weight * growth_factor, requires_grad=False)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            raise ValueError("Weights of the fully connected layer have become NaN.")
    

    def _train_forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: *NOT USED*
        @return
            output: returns the data after passing it through the layer
        """
        # Copy input -> calculate output -> update weights -> return output
        input_copy = input.clone().to(self.device).float()
        initial_input = input.clone().to(self.device).float()
        
        input_copy = self.fc(input_copy)
        output = self.inhibition(input_copy)
        self.update_weights(initial_input, output)
        # self.update_bias(input)
        self.weight_decay()
        
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
        # Copy input -> calculate output -> return output
        input_copy = input.clone().to(self.device).float()
        input_copy = self.fc(input_copy)
        output = self.inhibition(input_copy)
        return output