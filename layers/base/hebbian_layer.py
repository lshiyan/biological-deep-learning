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
                 weight_growth: WeightGrowth = WeightGrowth.LINEAR,
                 weight_decay: WeightDecay = WeightDecay.TANH,
                 bias_update: BiasUpdate = BiasUpdate.NO_BIAS
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
        self.decay: WeightDecay = weight_decay
        self.bias_update: BiasUpdate = bias_update


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
        if self.bias_update == BiasUpdate.NO_BIAS:
            return
        elif self.bias_update == BiasUpdate.SIMPLE:
            self._simple_bias_update(output)
        elif self.bias_update == BiasUpdate.HEBBIAN:
            self._hebbian_bias_update(output)
        else:
            raise ValueError("Update Bias type invalid.")
        

    def weight_decay(self) -> None:
        """
        METHOD
        Decays the overused weights and increases the underused weights using tanh functions.
        @param
            None
        @return
            None
        """
        # No weight decay
        if self.decay == WeightDecay.NO_DECAY:
            return

        # Determine the growth or decay factor
        if self.weight_growth == WeightGrowth.LINEAR:
            factor = self._linear_weight_decay() if self.decay == WeightDecay.TANH else self._simple_linear_weight_decay()
        elif self.weight_growth == WeightGrowth.SIGMOID:
            factor = self._sigmoid_weight_decay() if self.decay == WeightDecay.TANH else self._simple_sigmoid_weight_decay()
        else:
            raise ValueError(f"Invalid weight growth method: {self.weight_growth}")

        # Apply the decay or growth factor
        if self.decay == WeightDecay.TANH:
            updated_weights = self.fc.weight * factor
        elif self.decay == WeightDecay.SIMPLE:
            updated_weights = self.fc.weight - factor
        else:
            raise ValueError(f"Invalid weight decay method: {self.decay}")

        # Update the weights of the fully connected layer
        self.fc.weight = nn.Parameter(updated_weights, requires_grad=False)
    

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
        self.update_bias(output)
        self.weight_decay()
        
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
        # Copy input -> calculate output -> return output
        input_copy = input.clone().to(self.device).float()
        input_copy = self.fc(input_copy)
        output = self.inhibition(input_copy)
        return output