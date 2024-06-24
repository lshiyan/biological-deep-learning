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
        self.device_id = device                                             # WHY IS THIS NEEDED HERE? NOT NECESSARY AT ALL


    """
    Method to get layer with given name
    @param
        name (str) = name of layer to get
    @return
        layer (layer.NetworkLayer) = a layer of the network with searched name
    """
    def get_module(self, name):

    # STEP 1: loop through the network layeres
        for module_name, module in self.named_children():       
            # self.named_children() is a method provided by PyTorch’s nn.Module class 
            # It yields pairs of layer names (module_name) and the corresponding layer objects (module)

    # STEP 2: Layer matching and retrieval
            if name == module_name:
                return module
    

    """
    Method to set scheduler to all layers
    @param
    @return
        ___ (void) = no returns
    """
    def set_scheduler(self):

    # STEP 1: Iterate through child modules
        for module in self.children():
            # self.children() is a method from PyTorch’s nn.Module that iterates over all direct children modules (layers) of the current network module.
            
            module.set_scheduler()
            # The line above calls set_scheduler on each child module
    
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
        # Uses the get_module method to retrieve the layer by its name. 
        # This assumes the layer names are uniquely defined within the network.

        layer.set_scheduler(scheduler)
        # This calls the set_scheduler method on the retrieved layer, passing the scheduler as an argument. 
        # This sets the scheduler specifically for this layer, allowing it to update its parameters according to the scheduler’s logic.



    """
    Method to visualize the weights/features learned by each neuron during training
    @param
        path (Path) = path to print out result
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self, path, num, use):
        for module in self.children():
            module.visualize_weights(path, num, use)


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