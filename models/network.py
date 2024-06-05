from abc import ABC, abstractmethod
import torch.nn as nn

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
    Method to get layer with given name
    @param
        name (str) = name of layer to get
    @return
        layer (layer.NetworkLayer) = a layer of the network with searched name
    """
    def get_module(self, name):
        for module_name, module in self.named_children():
            if name == module_name:
                return module
    

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
    Method to set specific scheduler for specific layer
    @param
        name (str) = name of layer to set scheduler
        scheduler (layers.Scheduler) = scheduler to be set
    @return
        ___ (void) = no returns
    """
    def set_layer_scheduler(self, name, scheduler):
        layer = self.get_module(name)
        layer.set_scheduler(scheduler)


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
    Method to get number of active feature selectors
    @param
        beta (float) = cutoff value determining which neuron is active
    @return
        ___ (void) = no returns
    """
    def active_weights(self, beta):
        for name, module in self.name_children():
            print(f"{name}: {module.active_weights(beta)}")


    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data as a tensor
        clamped_output (TODO: ???) = parameter to clamp the output
    @return
        ___ (torch.Tensor) = processed data
    """ 
    # TODO: find a way to remove x and simply define an input processing layer that will create the necessary data to put into the network
    # NOTE: what is the use of clamped_out
    @abstractmethod  
    def forward(self, x, clamped_output=None):
        pass