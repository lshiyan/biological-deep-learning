from abc import ABC
import torch
import torch.nn as nn

from interfaces.layer import NetworkLayer


class Network(nn.Module, ABC):
    """
    INTERFACE
    Basic NN using 1 or more NetworkLayer -> Every NN for experiment must implement interface
    This will help with thesupport of multiple different networks
    
    @instance attr.
        device (str): device to which calculations will be made
    """
    def __init__(self, device: str) -> None:
        """
        CONSTRUCTOR METHOD
        @attr.
            device: device to which calculations will be made
        @return
            None
        """
        super().__init__()
        self.device = device


    def get_module(self, name: str) -> NetworkLayer:
        """
        METHOD
        Get layer with given name
        @param
            name: name of layer to get
        @return
            layer: a layer of the network with searched name
        """
        for module_name, module in self.named_children():       
            if name == module_name:
                return module


    def visualize_weights(self, path: str, num: int, use: str) -> None:
        """
        METHOD
        Visualize the weights/features learned by each neuron during training
        @param
            path: path to print out result
            num: which iteration is the visualization happening
            use: after what step is the visualization happening
        @return
            None
        """
        for name, module in self.named_children():
            module.visualize_weights(path, num, use, name)


    def active_weights(self, beta: float) -> dict[str:int]:
        """
        METHOD
        Get number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            module_active_weights: dictionary {str:int}
        """
        module_active_weights = {}
        
        for name, module in self.name_children():
            module_active_weights[name] = module.active_weights(beta)
        
        return module_active_weights


    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None):
        """
        METHOD
        Defines how an input data flows throw the network
        @param
            input: input data as a tensor
            clamped_output: tensor of true labels for classification
        @return
            ___ (torch.Tensor) = processed data
        """  
        raise NotImplementedError("This method is not implemented.")