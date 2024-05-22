import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


"""
Abstract class for a single layer of the ANN -> Every layer of the interface must implement interface
This will help with the support of multiple hidden layers inside the network
"""
class NetworkLayer (nn.Module, ABC):
    """
    Constructor method NetworkLayer
    @param
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        lamb (float) = lambda hyperparameter for latteral inhibition
        learning_rate (float) = how fast model learns at each iteration
        gamma (float) = decay factor -> factor to decay learning rate
        eps (float) = to avoid division by 0
    @attr.
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
        relu (fct) = ReLU function
        sigmoid (fct) = Sigmoid function
        softplus (fct) = Softplus function
        tanh (fct) = Tanh function
        softmax (fct) = Softmax function
    """
    def __init__(self, input_dimension, output_dimension, lamb=2, learning_rate=0.001, gamma=0.99, eps=10e-5):
        super ().__init__()
        self.input_dimension  = input_dimension
        self.output_dimension = output_dimension
        self.lamb = lamb
        self.alpha = learning_rate
        self.fc = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        self.scheduler = None
        self.eps = eps
        
        self.exponential_average = torch.zeros(self.output_dimension)
        self.gamma = gamma
        
        self.id_tensor = self.create_id_tensors(self)
        
        for param in self.fc.parameters():
            param = torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)

        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softplus=nn.Softplus()
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()

    """
    Sets scheduler current for layer
    @param
        scheduler (layers.Scheduler) = set the scheduler for current layer
    """
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        
    """
    Creates identity tensor
    """
    def create_id_tensors(self):
        id_tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity = torch.eye(i+1)
            padded_identity = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor
    
    """
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    @param
        row (int) = number of rows in display
        col (int) = number of columns in display
    """
    def visualize_weights(self, row, col):
        weight = self.fc.weight
        fig, axes = plt.subplots(row, col, figsize=(16, 8)) # FIXME: 16 and 8 are for classifying layer only -> what do these mean and put into parameters 
        for ele in range(self.num_neurons):  
            random_feature_selector = weight[ele]
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                    int(math.sqrt(self.fc.weight.size(1))))
            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
        plt.tight_layout()
        plt.show()
    
    """
    Defines the way the weights will be updated at each iteration of the training
    @param
        input (???) = ???
        output (???) = ???
    """
    # TODO: finish documentation when understand
    @abstractmethod
    def update_weights(self, input, output):
        pass

    """
    Defines the way the bias will be updated at each iteration of the training
    @param
        output (???) = ???
    """
    # TODO: finish documentation when understand
    @abstractmethod    
    def update_bias(self, output):
        pass
    
    """
    Feed forward
    """
    @abstractmethod
    def forward(self, x):
        pass