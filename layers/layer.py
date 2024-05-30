from abc import ABC, abstractmethod
import torch
import torch.nn as nn


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
    @return
        * Can't return *
    """
    def __init__(self, input_dimension, output_dimension, lamb=2, learning_rate=0.001, gamma=0.99, eps=10e-5):
        super ().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.lamb = lamb
        self.alpha = learning_rate
        self.fc = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        self.scheduler = None
        self.eps = eps
        
        self.exponential_average = torch.zeros(self.output_dimension)
        self.gamma = gamma
        
        self.id_tensor = self.create_id_tensors()
        
        for param in self.fc.parameters():
            param = torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)
        
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softplus=nn.Softplus()
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax()


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
    Sets scheduler current for layer
    """
    @abstractmethod
    def set_scheduler(self):
        pass


    """
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    """
    # TODO: find a way to automatically choose size of the plots, and how the plots will be arranged without needing to hard code it
    @abstractmethod
    def visualize_weights(self):
        pass
    

    """
    Defines the way the weights will be updated at each iteration of the training
    @param
        input_data (???) = ???
        output_data (???) = ???
        clamped_output (???) = ???
    """
    # TODO: finish documentation when understand
    # NOTE: what is clamped_output
    @abstractmethod
    def update_weights(self, input, output, clamped_output):
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
    @param
        x (torch.Tensor) = inputs into the layer
        clamped_output (???) = ???
    """
    # TODO: finish documentation when understand
    # NOTE: what is clamped_output?
    @abstractmethod
    def forward(self, x, clamped_output):
        pass

    """
    Counts the number of active feature selectors (above a certain cutoff beta).
    @param
        beta (float) = cutoff value determining which neuron is active and which is not
    """
    @abstractmethod
    def active_weights(self, beta):
        pass