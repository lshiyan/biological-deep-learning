from abc import ABC, abstractmethod
import torch.nn as nn 

"""
Interface for all networks to be created
"""
class Network(nn.Module, ABC):
    """
    Constructor method
    @attr.
        __layers (dict {str:nn.Module}) = list of layers of the network
    @pram
    @return
        * Can't return *
    """
    def __init__(self):
        super().__init__()


    """
    Function returning layer with given name
    @param
        name (str) = name of layer to get
    @return
        layer (layer.NetworkLayer) = a layer of the network with searched name
    """
    def get_layer(self, name):
        layers = self.named_children()

        for layer_name, layer in layers:
            if name == layer_name:
                return layer
        return None
    

    """
    Method to set scheduler to all layers
    @param
    @return
        ___ (void) = no returns
    """
    # NOTE: What use is this???
    def set_scheduler(self):
        for module in self.children():
            module.set_scheduler()

    
    """
    Method to visualize the weights/features learned by each neuron during training
    @param
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self):
        for module in self.children():
            module.visualize_weights()


    """
    Returns number of active feature selectors
    @param
        beta (float) = cutoff value determining which neuron is active
    @return
        ___ (void) = no returns
    """
    def active_weights(self, beta):
        for module in self.children():
            module.active_weights(beta)


    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data as a tensor
        clamped_output (???) = parameter to clamp the output   # TODO: Figure out what clamped_output is used for
    @return
        ___ (torch.Tensor) = processed data
    """ 
    # TODO: find a way to remove x and simply define an input processing layer that will create the necessary data to put into the network
    # NOTE: what is the use of clamped_out
    @abstractmethod  
    def forward(self, x, clamped_output=None):
        pass