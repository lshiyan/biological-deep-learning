import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.hidden_layer import HiddenLayer
from utils.experiment_constants import ActivationMethods, BiasUpdate, Focus, LateralInhibitions, LearningRules, ParamInit, WeightDecay, WeightGrowth


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
    
    #################################################################################################
    # Constructor Method
    #################################################################################################
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
                         init)
        self.inhibition_rule: LateralInhibitions = inhibition_rule
        self.learning_rule: LearningRules = learning_rule
        self.weight_growth: WeightGrowth = weight_growth
        self.bias_update: BiasUpdate = bias_update
        self.focus: Focus = focus
        self.activation_method: ActivationMethods = activation
        
        self.gamma: float = gamma
        self.lamb: float = lamb
        self.eps: float = eps
        self.sigmoid_k: float = sigmoid_k
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension).to(self.device)

        self.normalized_weights: torch.Tensor = self.normalize(self.fc.weight).to(self.device)
        


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
        elif self.inhibition_rule == LateralInhibitions.MAX_INHIBITION:
            return self._max_inhibition(input)
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
        elif self.weight_growth == WeightGrowth.EXPONENTIAL:
            function_derivative = self._exponential_function()
        else:
            raise NameError("Unknown weight growth rule.")
            
        # Weight Update
        delta_weight: torch.Tensor = (self.lr * calculated_rule * function_derivative).to(self.device)
        updated_weight: torch.Tensor = torch.add(self.fc.weight, delta_weight)
        self.fc.weight = nn.Parameter(updated_weight, requires_grad=False)
        
        # Normalized Weight Update
        self.normalized_weights = self.normalize(updated_weight).to(self.device)
        

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
    
    

    #################################################################################################
    # Training and Evaluation Methods
    #################################################################################################
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
        # Calculate activation -> Calculate inhibition -> Update weights -> Update bias -> Rreturn output
        activations: torch.Tensor = self.activation(input)
        output: torch.Tensor = self.inhibition(activations)
        self.update_weights(input, output)
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
        output: torch.Tensor = self.inhibition(activations)
        
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
        return F.linear(input, self.normalized_weights, bias=self.fc.bias)
    
    
    
    #################################################################################################
    # Different Inhibition Methods
    #################################################################################################
    def _relu_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates ReLU lateral inhibition
        Inhibition: x = [ ReLU(x) / max(x) ] ^ lamb
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # Get ReLU activation function
        relu: nn.ReLU = nn.ReLU()
        
        # Compute ReLU and lateral inhibition
        relu_input: torch.Tensor = relu(input)
        max_ele: float = torch.max(input).item()
        output: torch.Tensor =  ((relu_input / max_ele) ** self.lamb).to(self.device) if max_ele != 0 else torch.zeros(relu_input.size())
        
        return output
    
    
    def _max_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates ReLU lateral inhibition
        Inhibition: x = [ abs(x) / abs(x) ] ^ lamb * proper sign
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        sign: torch.Tensor = torch.where(input>=0, 1, -1)    
        abs_input: torch.Tensor = torch.abs(input)
        max_ele: float = torch.max(abs_input).item()
        output: torch.Tensor = (((abs_input / max_ele) ** self.lamb) * sign).to(self.device)
        
        return output
             
    
    def _exp_softmax_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates exponential softmax (Modern Hopfield) lateral inhibition
        Inhibition: x = Softmax((x - max(x)) * lamb)
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # Computes exponential softmax with numerical stabilization
        max_ele: float = torch.max(input).item()
        output: torch.Tensor = F.softmax((input - max_ele) * self.lamb, dim=-1).to(self.device)
        return output
    
        
    def _wta_inhibition(self, input:torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates k-winners-takes-all lateral inhibition
        Inhibition: x = x if x > threshold else 0
        @param
            input: input to layer
            threshold: amount to be counted as activated
        @return
            output: activation after lateral inhibition
        """
        # Computes winner-takes-all inhibition
        max_ele: float = torch.max(input).item()
        output: torch.Tensor = torch.where(input>=max_ele, 1.0, 0.0).to(self.device)

        return output
    
    
    def _gaussian_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates gaussian lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        mu: float = torch.max(torch.mean(input)).item() 
        sigma: float = 1 / math.sqrt(2 * math.pi)
        gaussian: torch.Tensor = ((1 / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-self.lamb * ((input - mu) / sigma) ** 2)).to(self.device)
        output: torch.Tensor = gaussian # NOTE: should i multiply gaussian by input?
        
        return output

    
    def _norm_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates devisive normalization lateral inhibition
        Inhibition: x = ReLU(x) / sum(ReLU(x))
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # Get ReLU activation function
        relu: nn.ReLU = nn.ReLU()
        
        # Compute ReLU and lateral inhibition
        activation: torch.Tensor = relu(input).to(self.device)
        sum: float = input.sum().item()
        output: torch.Tensor =  (activation / sum).to(self.device)
        
        return output
    
    
    #################################################################################################
    # Different Weight Updates Methods
    #################################################################################################
    def _hebbian_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Computes Hebbian Leanring Rule.
        Rule: delta Wij = (y * x)ij 
        @param
            input: the input of the layer
            output: the output of the layer
        @return
            computed_rule: computed hebbian rule
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        computed_rule: torch.Tensor = torch.einsum("i, j -> ij", y, x).to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule

    
    def _sanger_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Computes Sanger's Rule
        Rule: delta Wij = ((y * x)ij - yi * SUM(Wkj * yk, k=1 to i))
        @param
            input: the input of the layer
            output: the output of the layer
        @return
            computed_rule: computed sanger's rule
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.einsum("i, j -> ij", y, x).to(self.device)

        # Retrieve initial weights
        id_tensor: torch.Tensor = self.create_id_tensors(self.output_dimension).to(self.device)
        weights: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
        wi_norm: torch.Tensor = self.get_norm(weights).to(self.device)
        wk_norm: torch.Tensor = self.get_norm(weights).to(self.device)
        wk_norm = torch.ones(wk_norm.size()) / wk_norm
        w_ratio: torch.Tensor = torch.einsum('a, b -> ab', wi_norm, wk_norm).to(self.device)
        w_ratio = w_ratio.expand_as(id_tensor)
        w_ratio = w_ratio * id_tensor

        # Calculate Sanger's Rule
        A: torch.Tensor = torch.einsum('kj, lkm, m, l -> lj', weights, w_ratio, y, y).to(self.device)
        computed_rule: torch.Tensor = (outer_prod - A).to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule


    def _fully_orthogonal_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Update weights using Fully Orthogonal Rule.
        Rule: delta Wij = ((y * x)ij - yi * SUM(W * yk))
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
        outer_prod: torch.Tensor = torch.einsum("i, j -> ij", y, x).to(self.device)

        # Retrieve initial weights
        weights: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
        wi_norm: torch.Tensor = self.get_norm(weights).to(self.device)
        wk_norm: torch.Tensor = self.get_norm(weights).to(self.device)
        wk_norm = torch.ones(wk_norm.size()) / wk_norm
        w_ratio: torch.Tensor = torch.einsum('a, b -> ab', wi_norm, wk_norm).to(self.device)

        # Calculate Fully Orthogonal Rule
        norm_term: torch.Tensor = torch.einsum("i, ij, k -> kj", y, weights, y)
        norm_term = norm_term / w_ratio

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
        Derivative: 1/K * (K - Wij) * Wij or 1/K * (K - Wi:) * Wi:
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
            derivative = (1 / self.sigmoid_k) * (self.sigmoid_k - norm) * norm
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
            derivative = current_weights
        elif self.focus == Focus.NEURON:
            norm: torch.Tensor = self.get_norm(self.fc.weight)
            derivative = norm
        else:
            raise ValueError("Invalid focus type.")
        
        return derivative
        

        
    #################################################################################################
    # Different Weights Growth for Weight Updates
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
        """
        METHOD
        Simple linear weight decay
        Decay: linear_derivative * eps * (ai / a - 1)
        @param
            None
        @return
            decay_weight: amount to decay the weights by
        """
        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()
        
        # Compute simple weight decay
        decay_weight: torch.Tensor = (self.eps * (self.exponential_average / average - 1)).unsqueeze(1).to(self.device)
        decay_weight = self._linear_function() * decay_weight
        
        return decay_weight
    
    
    def _simple_sigmoid_weight_decay(self) -> torch.Tensor:
        """
        METHOD
        Simple linear weight decay
        Decay: sigmoid_derivative * eps * (ai / a - 1)
        @param
            None
        @return
            decay_weight: amount to decay the weights by
        """
        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()
        
        # Compute simple weight decay
        decay_weight: torch.Tensor = (self.eps * (self.exponential_average / average - 1)).unsqueeze(1).to(self.device)
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
        
        
        
    #################################################################################################
    # Static Methods
    #################################################################################################
    @staticmethod
    def get_norm(weights: torch.Tensor) -> torch.Tensor:
        norm: torch.Tensor = torch.norm(weights, p=2, dim=-1, keepdim=True)
        return norm
    
    
    @staticmethod
    def normalize(weights: torch.Tensor) -> torch.Tensor:
        norm: torch.Tensor = HebbianLayer.get_norm(weights)
        normalized_weights: torch.Tensor = weights / norm
        
        return normalized_weights