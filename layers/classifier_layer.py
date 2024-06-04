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
        gamma (float) = decay factor -> factor to decay learning rate
        eps (float) = to avoid division by 0
    @attr.
        PARENT ATTR.
            input_dimension (int) = number of inputs into the layer
            output_dimension (int) = number of outputs from layer
            lamb (float) = lambda hyperparameter for latteral inhibition
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            scheduler (layers.Scheduler) = scheduler for current layer
            eps (float) = to avoid division by 0
            exponential_average (torch.Tensor) = 0 tensor to keep track of exponential averages
            gamma (float) = decay factor -> factor to decay learning rate
            id_tensor (torch.Tensor) = id tensor of layer
        OWN ATTR.
    @return
        ___ (layers.ClassifierLayer) = returns instance of ClassifierLayer
    """
    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, class_lr=0.001, gamma=0.99, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, lamb, class_lr, gamma, eps)    
        self.device_id = device_id 
    
    """
    Defines the way the weights will be updated at each iteration of the training.
    The method computes the outer product of the softmax probabilities of the outputs and the inputs. 
    This product is scaled by the learning rate and used to adjust the weights. 
    The weights are then normalized to ensure stability.
    @param
        input (torch.Tensor): The input tensor to the layer before any transformation.
        output (torch.Tensor): The output tensor of the layer before applying softmax.
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
            u_times_y  =torch.mul(u,y)
            A = outer_prod  -self.fc.weight * (u_times_y.unsqueeze(1))
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
        

    """
    Defines the way the biases will be updated at each iteration of the training
    It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.
    The method applies an exponential decay to the biases, which is modulated by the output probabilities,
    and scales the update by the learning rate. 
    The biases are normalized after the update.
    @param
        output (torch.Tensor): The output tensor of the layer before applying softmax.
    @return
        ___ (void) = no returns
    """
    def update_bias(self, output):
        y = output.clone().detach().squeeze() # Output tensor probabilities after softmax
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y)-1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability.
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
    
    """
    Feed forward
    @param
        x (torch.Tensor) = data after going through hebbian layer
        clamped_output (???) = ???
    @retrun
        x (torch.Tensor) = data after going through classification layer
    """
    # NOTE: what is clamped_output
    def forward(self, x, clamped_output=None):
        input_copy = x.clone()
        x = self.fc(x)
        self.update_weights(input_copy, x, clamped_output)
        # self.update_bias(x)
        x = self.softmax(x)
        return x
    
    """
    Counts the number of active feature selectors (above a certain cutoff beta).
    @param
        beta (float) = cutoff value determining which neuron is active and which is not
    @return
        active.nonzero().size(0) (int) = nunmber of active weights
    """
    def active_weights(self, beta):
        weights = self.fc.weight
        active = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)


    """
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    @param
        result_path (Path) = path to folder where results will be printed
    @return
        ___ (void) = no returns
    """
    # TODO: make the size and presentation of the plots less hard coded AKA replace the 2, 5, 16, 8 with variables
    def visualize_weights(self, result_path):
        weight = self.fc.weight
        fig, axes = plt.subplots(2, 5, figsize=(16, 8))
        for ele in range(2*5):  
            random_feature_selector = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // 5, ele % 5]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')

            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device_id)

        file_path = result_path + '/classifierlayerweights.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.show()

    """
    Sets the scheduler for this layer
    @param
    @return
        ___ (void) = no returns
    """
    # TODO: define this function
    def set_scheduler(self):
        pass