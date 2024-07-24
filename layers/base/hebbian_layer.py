from typing import Optional
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
        HiddenLayer ATTR.
            exponential_average (torch.Tensor): tensor to keep track of exponential averages
            id_tensor (torch.Tensor): id tensor of layer
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            sigmoid_k (float): constant for sigmoid wieght growth updates
        OWN ATTR.
            inhibition_rule (LateralInhibitions): which inhibition to be used
            learning_rule (LearningRules): which learning rule to use
            weight_growth (WeightGrowth): which function type should the weight updates follow
            weight_decay (WeightDecay): which weight decay to use
            bias_update (BiasUpdate): which bias update rule to follow
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
            inhibition_rule: which inhibition to be used
            learning_rule: which learning rule to use
            weight_growth: which function type should the weight updates follow
            weight_decay: which weight decay to use
            bias_update: which bias update rule to follow
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
            outputs of inhibiton
        """ 
        if self.inhibition_rule == LateralInhibitions.RELU_INHIBITION:
            return self._relu_inhibition(input)
        elif self.inhibition_rule == LateralInhibitions.EXP_SOFTMAX_INHIBITION:
            return self._exp_softmax_inhibition(input)
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
        else:
            raise NameError("Unknown learning rule.")
        
        if self.weight_growth == WeightGrowth.LINEAR:
            function_derivative = self._linear_function()
        elif self.weight_growth == WeightGrowth.SIGMOID:
            function_derivative = self._sigmoid_function()
        else:
            raise NameError("Unknown weight growth rule.")
            
        # Weight Update
        delta_weight: torch.Tensor = (self.lr * calculated_rule * function_derivative).to(self.device)
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        

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
        elif self.bias_update == BiasUpdate.SIMPLE:
            self._simple_bias_update(output)
        elif self.bias_update == BiasUpdate.HEBBIAN:
            self._hebbian_bias_update(output)
        else:
            raise ValueError("Update Bias type invalid.")
        

    def weight_decay(self) -> None:
        """
        METHOD
        Decays weights according to rule
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
            updated_weights = (self.fc.weight * factor).to(self.device)
        elif self.decay == WeightDecay.SIMPLE:
            updated_weights = (self.fc.weight - factor).to(self.device)
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
        # Calculate output -> Update weights -> Update bias -> Decay weights -> return output
        input_copy = input.clone().to(self.device).float()
        
        output = self.inhibition(self.fc(input_copy))
        self.update_weights(input_copy, output)
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
        # Calculate output -> Return output
        input_copy = input.clone().to(self.device).float()
        output = self.inhibition(self.fc(input_copy))
        return output