import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from layers.layer import NetworkLayer


"""
Class defining the functionality of the classification layer
"""
class ClassifierLayer(NetworkLayer):
    """
    Constructor method NetworkLayer
    @param
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        lamb (float) = lambda hyperparameter for latteral inhibition
        class_lr (float) = how fast model learns at each iteration
        eps (float) = to avoid division by 0
    @attr.
        PARENT ATTR.
            input_dimension (int) = number of inputs into the layer
            output_dimension (int) = number of outputs from layer
            device_id (int) = the device that the module will be running on
            lamb (float) = lambda hyperparameter for latteral inhibition
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            eps (float) = to avoid division by 0
        OWN ATTR.
    @return
        ___ (layers.ClassifierLayer) = returns instance of ClassifierLayer
    """
    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, class_lr=0.001, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, device_id, lamb, class_lr, eps)    
    

    """
    Defines the way the weights will be updated at each iteration of the training.
    The method computes the outer product of the softmax probabilities of the outputs and the inputs. 
    This product is scaled by the learning rate and used to adjust the weights. 
    The weights are then normalized to ensure stability.
    @param
        input (torch.Tensor): The input tensor to the layer before any transformation.
        output (torch.Tensor): The output tensor of the layer before applying softmax.
        clamped_output (torch.Tensor): one-hot encode of true labels
    @return
        ___ (void) = no returns
    """
    def update_weights(self, input, output, clamped_output=None):
        # Detach and squeeze tensors to remove any dependencies and reduce dimensions if possible.
        u = output.clone().detach().squeeze() # Output tensor after layer but before activation
        x = input.clone().detach().squeeze() # Input tensor to layer
        y = torch.softmax(u, dim=0) # Apply softmax to output tensor to get probabilities
        A = None

        if clamped_output != None:
            outer_prod = torch.outer(clamped_output-y,x)
            u_times_y = torch.mul(u,y)
            A = outer_prod - self.fc.weight * (u_times_y.unsqueeze(1))
        else:
            # Compute the outer product of the softmax output and input.
            A = torch.outer(y,x) # Hebbian learning rule component

        # Adjust weights by learning rate and add contribution from Hebbian update.
        A = self.fc.weight + self.alpha * A

        # Normalize weights by the maximum value in each row to stabilize the learning.
        weight_maxes = torch.max(A, dim=1).values
        self.fc.weight = nn.Parameter(A/weight_maxes.unsqueeze(1), requires_grad=False)

        # Zero out the first column of weights -> this is to prevent the first weight from learning everything
        self.fc.weight[:, 0] = 0
        

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y)-1
        A = (1 - self.alpha) * self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes = torch.max(A, dim=0).values
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
        softmax = nn.Softmax(dim=1)
        input_copy = input.clone()
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
        row = 0
        col = 0

        root = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Gets the weights and create heatmap
        weight = self.fc.weight
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')

            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device_id)

        file_path = result_path + f'/classification/classifierlayerweights-{num}-{use}.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    
    def active_weights(self, beta):
        """
        METHOD
        Get number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        weights = self.fc.weight
        active = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)