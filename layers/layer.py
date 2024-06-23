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
        eps (float) = to avoid division by 0
    @attr.
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        device_id (int) = the device that the module will be running on
        lamb (float) = lambda hyperparameter for latteral inhibition
        alpha (float) = how fast model learns at each iteration
        fc (fct) = function to apply linear transformation to incoming data
        eps (float) = to avoid division by 0
    @return
        * Can't return *
    """
    def __init__(self, input_dimension, output_dimension, device_id, lamb=1, learning_rate=0.005, eps=0.01):
        super ().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.device_id = device_id
        self.lamb = lamb
        self.alpha = learning_rate
        self.fc = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        self.eps = eps
        
        for param in self.fc.parameters():
            torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)


    """
    Method to create an identity tensor
    @param
    @return
        id_tensor (torch.Tensor) = 3D tensor with increasing size of identify matrices
    """
    def create_id_tensors(self):
        id_tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity = torch.eye(i+1)
            padded_identity = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor


    """
    Method to vizualize the weight/features learned by neurons in this layer using a heatmap
    @param
        result_path (Path) = path to folder where results will be printed
    @return
        ___ (void) = no returns
    """
    @abstractmethod
    def visualize_weights(self, result_path, num, use):
        pass
    

    """
    Method to define the way the weights will be updated at each iteration of the training
    @param
        input (torch.Tensor) = the inputs into the layer
        output (torch.Tensor) = the output of the layer
        clamped_output (torch.Tensor) = true labels
    @return
        ___ (void) = no returns
    """
    @abstractmethod
    def update_weights(self, input, output, clamped_output=None):
        pass


    """
    Method to define the way the bias will be updated at each iteration of the training
    @param
        output (torch.Tensor) = the output of the layer
    @return
        ___ (void) = no returns
    """
    @abstractmethod 
    def update_bias(self, output):
        pass
    

    """
    Method that defines how an input data flows throw the network when training
    @param
        x (torch.Tensor) = input data into the layer
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the layer
    """
    @abstractmethod
    def _train_forward(self, x, clamped_output=None):
        pass
    
    
    """
    Method that defines how an input data flows throw the network when testing
    @param
        x (torch.Tensor) = input data into the layer
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the layer
    """
    @abstractmethod
    def _eval_forward(self, x):
        pass


    """
    Method to get number of active feature selectors
    @param
        beta (float) = cutoff value determining which neuron is active
    @return
        ___ (void) = no returns
    """
    @abstractmethod
    def active_weights(self, beta):
        pass

    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data into the layer
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the layer
    """
    def forward(self, x, clamped_output=None):
        if self.training:
            x = self._train_forward(x, clamped_output)
        else:
            x = self._eval_forward(x)
        return x
    