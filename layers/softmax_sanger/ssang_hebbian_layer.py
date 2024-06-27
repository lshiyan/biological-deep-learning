import torch
import torch.nn as nn
from layers.hidden_layer import HiddenLayer


class SSangHebbianLayer(HiddenLayer):
    """
    CLASS
    Defining the functionality of the base hebbian layer
    @instance attr.
        NetworkLayer ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): the device that the module will be running on
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        HiddenLayer ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            id_tensor (torch.Tensor): id tensor of layer
        OWN ATTR.
    """
    def __init__(self, input_dimension: int, 
                 output_dimension: int, 
                 device: str, 
                 lamb: float = 1, 
                 learning_rate: float = 0.005, 
                 gamma: float = 0.99, 
                 eps: float = 0.01) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            lamb: lambda hyperparameter for lateral inhibition
            learning_rate: how fast model learns at each iteration
            gamma: affects exponentialaverages updates
            eps: affects weight decay updates
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device, learning_rate, lamb, gamma, eps)

    
    def inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Choice of inhibition to be used.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            inputs after inhibition
        """
        return self._softmax_inhibition(input)
    
    
    def update_weights(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Sanger's Rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        self._linear_sanger_rule(input, output)
        

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
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: float = torch.max(A, dim=0).values
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
        tanh: nn.Tanh = nn.Tanh()

        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()

        # Gets ratio vs mean
        A: torch.Tensor = self.exponential_average / average

        # calculate the growth factors
        growth_factor_positive: torch.Tensor = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative: torch.Tensor = torch.reciprocal(growth_factor_positive)

        # Update the weights depending on growth factor
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")
    

    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: *NOT USED*
        @return
            output: returns the data after passing it throw the layer
        """
        # Copy input -> calculate output -> update weights -> return output
        input_copy = input.clone().to(self.device).float()
        input = input.to(self.device)
        input = self.fc(input)
        output = self.inhibition(input)
        self.update_weights(input_copy, output)
        #self.update_bias(input)
        self.weight_decay()
        return output
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Define how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            output: returns the data after passing it throw the layer
        """
        # Copy input -> calculate output -> return output
        input = input.to(self.device)
        input = self.fc(input)
        output = self.inhibition(input)
        return output