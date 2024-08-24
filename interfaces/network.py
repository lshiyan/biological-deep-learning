from abc import ABC
from typing import Optional
import torch
import torch.nn as nn
from utils.experiment_constants import LayerNames


class Network(nn.Module, ABC):
    """
    INTERFACE
    Defines an ANN that uses 1 or more NetworkLayers -> Every ANN for experiment must implement interface
    @instance attr.
        name (str): name of model
        device (str): device to which calculations will be made
    """
    def __init__(self, name: str, device: str) -> None:
        """
        CONSTRUCTOR METHOD
        @attr.
            name: name of network
            device: device that will be used
        @return
            None
        """
        super().__init__()
        self.name: str = name
        self.device: str = device


    def get_module(self, lname: LayerNames) -> nn.Module:
        """
        METHOD
        Returns layer with given name
        @param
            name: name of layer to get
        @return
            layer: layer of the network with searched name
        """
        for layer_name, layer in self.named_children():      
            if lname.name.upper() == layer_name.upper():
                return layer
        raise NameError(f"There are no layer named {lname.name.upper()}.")


    def visualize_weights(self, path: str, num: int, use: str, coloured: bool) -> None:
        """
        METHOD
        Visualizes the weights/features learned by each neuron during training
        @param
            path: path to print out result
            num: which iteration is the visualization happening
            use: after what step is the visualization happening
            colored: whether or not it is a coloured image
        @return
            None
        """
        for name, module in self.named_children():
            if not coloured:
                module.visualize_weights(path, num, use, name.lower().capitalize())
            if coloured: 
                module.visualize_colored_weights(path, num, use, name.lower().capitalize())

    def active_weights(self, beta: float) -> dict[str, int]:
        """
        METHOD
        Returns number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            module_active_weights: dictionary {str:int}
        """
        module_active_weights: dict[str, int] = {}
        
        for name, module in self.name_children():
            module_active_weights[name.lower()] = module.active_weights(beta)
        
        return module_active_weights


    def forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor: 
        raise NotImplementedError("This method is not implemented.")