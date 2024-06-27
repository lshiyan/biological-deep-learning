import torch
import torch.nn as nn
from layers.output_layer import OutputLayer


class RYZZClassificationLayer(OutputLayer):
    """
    CLASS
    Defining the functionality of the base classification layer
    
    @instance attr.
        NetworkLayer ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): the device that the module will be running on
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
    """
    def __init__(self, input_dimension: int,
                 output_dimension: int, 
                 device: str, 
                 learning_rate: float = 0.005) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: the device that the module will be running on
            learning_rate: how fast model learns at each iteration
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device, learning_rate)    
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None, include_first: bool = True) -> None:
        """
        METHOD
        Defines the way the weights will be updated at each iteration of the training.
        @param
            input: The input tensor to the layer before any transformation.
            output: The output tensor of the layer before applying softmax.
            clamped_output: one-hot encode of true labels
            include_first: determine if first neuron is included in classification
        @return
            None
        """
        # Detach and squeeze tensors to remove any dependencies and reduce dimensions if possible.
        u: torch.Tensor = output.clone().detach().squeeze() # Output tensor after layer but before activation
        x: torch.Tensor = input.clone().detach().squeeze() # Input tensor to layer
        y: torch.Tensor = torch.softmax(u, dim=0) # Apply softmax to output tensor to get probabilities
        A: torch.Tensor = None

        if clamped_output != None:
            outer_prod: torch.Tensor = torch.outer(clamped_output-y,x)
            u_times_y: torch.Tensor = torch.mul(u,y)
            A = outer_prod - self.fc.weight * (u_times_y.unsqueeze(1))
        else:
            A = torch.outer(y, x)

        # STEP 3: Adjust weights
        weights = self.fc.weight + self.lr * A
        # The weights are adjusted by adding the scaled update A to the current weights.


        # Normalize weights by the maximum value in each row to stabilize the learning.
        weight_maxes: torch.Tensor = torch.max(weights, dim=1).values
        self.fc.weight = nn.Parameter(weights/weight_maxes.unsqueeze(1), requires_grad=False)

        # Zero out the first column of weights -> this is to prevent the first weight from learning everything
        if not include_first: self.fc.weight[:, 0] = 0
        

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        y: torch.Tensor = output.clone().detach().squeeze()
        exponential_bias: torch.Tensor = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = torch.mul(exponential_bias, y)-1
        A = (1 - self.lr) * self.fc.bias + self.lr * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: torch.Tensor = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
    

    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None, include_first: bool = True) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            output: returns the data after passing it throw the layer
        """
        softmax: nn.Softmax = nn.Softmax(dim=1)
        input_copy: torch.Tensor = input.clone()
        input = self.fc(input)
        self.update_weights(input_copy, input, clamped_output, include_first=True)
        # self.update_bias(input)
        output = softmax(input)
        return output
    
    
    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            input: returns the data after passing it throw the layer
        """
        softmax = nn.Softmax(dim=1)
        input = self.fc(input)
        output = softmax(input)
        return output