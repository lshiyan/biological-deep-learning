from abc import ABC, abstractmethod
import torch.nn as nn 
import os

"""
Interface for all networks to be created
"""
class Network(nn.Module, ABC):
    """
    Constructor method
    @attr.
        device_id (int) = id of the gpu that the model will be running in
    @pram
        device (int) = id of the gpu that the model will be set to
    @return
        * Can't return *
    """
    def __init__(self, device):
        super().__init__()
        self.device_id = device


    """
    Function returning layer with given name
    @param
        name (str) = name of layer to get
    @return
        layer (layer.NetworkLayer) = a layer of the network with searched name
    """
    def get_layer(self, name):
        for layer_name, layer in self.named_children():
            if name == layer_name:
                return layer
    

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
        path (Path) = path to print out result
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self, path):
        for module in self.children():
            module.visualize_weights(path)







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