import math
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from layers.layer import NetworkLayer


class ClassifierLayer(NetworkLayer):
    """
    CLASS
    Defining the functionality of the classification layer
    
    @instance attr.
        PARENT ATTR.
            input_dimension (int) = number of inputs into the layer
            output_dimension (int) = number of outputs from layer
            device_id (int) = the device that the module will be running on
            lamb (float) = lambda hyperparameter for latteral inhibition
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            eps (float) = to avoid division by 0
        OWN ATTR.
    """
    def __init__(self, input_dimension: int,
                 output_dimension: int, 
                 device_id: str, 
                 lamb: float = 1, 
                 class_lr: float = 0.005, 
                 eps: float = 0.01) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device_id: the device that the module will be running on
            lamb: lambda hyperparameter for latteral inhibition
            class_lr: how fast model learns at each iteration
            eps: to avoid division by 0

        @return
            None
        """
        super ().__init__(input_dimension, output_dimension, device_id, lamb, class_lr, eps)    
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None) -> None:
        """
        METHOD
        Defines the way the weights will be updated at each iteration of the training.
        @param
            input: The input tensor to the layer before any transformation.
            output: The output tensor of the layer before applying softmax.
            clamped_output: one-hot encode of true labels
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
            A = torch.outer(post_activation_output, input_value)  # Hebbian learning rule component
            # If clamped_output is not provided, A is simply the outer product of y and x.


    # STEP 3: Adjust weights
        A = self.fc.weight + self.alpha * A
        # The weights are adjusted by adding the scaled update A to the current weights.


        # Normalize weights by the maximum value in each row to stabilize the learning.
        weight_maxes: torch.Tensor = torch.max(A, dim=1).values
        self.fc.weight = nn.Parameter(A/weight_maxes.unsqueeze(1), requires_grad=False)

        # Zero out the first column of weights -> this is to prevent the first weight from learning everything
        # self.fc.weight[:, 0] = 0
        

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
        A = (1 - self.alpha) * self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: torch.Tensor = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
    

    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            input: returns the data after passing it throw the layer
        """
        softmax: nn.Softmax = nn.Softmax(dim=1)
        input_copy: torch.Tensor = input.clone()
        input = self.fc(input)
        self.update_weights(input_copy, input, clamped_output)
        # self.update_bias(input)
        input = softmax(input)
        return input
    
    
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
        input = softmax(input)
        return input


    def visualize_weights(self, result_path: str, num: int, use: str) -> None:
        """
        METHOD
        Vizualize the weight/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
        @return
            None
        """
        # Find value for row and column
        row: int = 0
        col: int = 0

        root: int = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Gets the weights and create heatmap
        weight: nn.parameter.Parameter = self.fc.weight
        fig: matplotlib.figure.Figure = None
        axes: np.ndarray = None
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector: torch.Tensor = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap: torch.Tensor = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')

            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device_id)

        file_path: str = result_path + f'/classification/classifierlayerweights-{num}-{use}.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    
    def active_weights(self, beta: float) -> int:
        """
        METHOD
        Get number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        weights: nn.parameter.Parameter = self.fc.weight
        active: torch.Tensor = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)