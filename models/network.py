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
        layers (dict) = dictionary where {key(str):value(nn.Module)}
    """
    def __init__(self):
        super().__init__()
        self.__layers = {}


    """
    Adds a layer to the network
    @param
        name (str) = name of the layer
        layer (layers.NetworkLayer) = layer that is being added
    """
    def add_layer(self, name, layer):
        if name not in self.__layers.keys():
            self.__layers[name] = layer
        else:
            print("Layer already exists.")


    """
    Function returning layer with given name
    @param
        name (str) = name of layer to get
    """
    def get_layer(self, name):
        return self.__layers[name]


    """
    Returns an iterator for only the layers within the network
    """
    def layers(self):
        return self.__layers.values()


    """
    Returns an iterator for name, layer for all layers within the network
    """
    def named_layers(self):
        return self.__layers.items()
    

    """
    Method to set scheduler to all layers
    """
    def set_scheduler(self):
        for module in self.layers():
            module.set_scheduler()

    
    """
    Method to visualize the weights/features learned by each neuron during training
    """
    def visualize_weights(self):
        for module in self.layers():
            module.visualize_weights()


    """
    Returns number of active feature selectors
    @param
        beta (float) = cutoff value determining which neuron is active
    """
    def active_weights(self, beta):
        for module in self.layers():
            module.active_weights(beta)


    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data as a tensor
        clamped_output (???) = parameter to clamp the output   # WTV this means
    """   
    @abstractmethod
    def forward(self, x, clamped_output=None):
        pass