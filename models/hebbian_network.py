##############################################################################
# PART 1: Imports and Timer Initialization
##############################################################################

# Here:
    # Import necessary libraries and modules

import torch.nn as nn
import torch
from layers.input_layer import InputLayer
from layers.hebbian_layer import HebbianLayer
from layers.classifier_layer import ClassifierLayer
from models.network import Network 

class HebbianNetwork(Network): # Inherits from the Network base class
    """
    Constructor method
    @attr.
        PARENT ATTR.
            device_id (int) = id of the gpu that the model will be running in
        OWN ATTR.
            input_dim (int) = number of inputs
            heb_dim (int) = number of neurons in hebbina layer
            output_dimension (int) = number of output neurons
            heb_param (dict {str:float}) = dictionary with all the hyperparameters for the hebbian layer
                - lr (float) = learning rate of hebbian layer
                - lamb (float) = hyperparameter for lateral neuron inhibition
                - gam (float) = factor to decay learning rate of hebbian layer
            cla_param (dict {str:float}) = dictionary with all the hyperparameters for the classification layer
                - lr (float) = learning rate of classification layer
                - lamb (float) = hyperparameter for lateral neuron inhibition
            eps (float) = small value to avoid 0 division
    @pram
        args (argparse.ArgumentParser) = arguments passed from command line
    @return
        ___ (models.HebianNetwork) = new instance of a HebbianNetwork
    """
    def __init__(self, args):
        super().__init__(args.device_id)

        # Dimension of each layer
        self.input_dim = args.input_dim
        self.heb_dim = args.heb_dim
        self.output_dim = args.output_dim

        # Hebbian layer hyperparameters
        self.heb_param = {}
        self.heb_param["lamb"] = args.heb_lamb
        self.heb_param["gam"] = args.heb_gam

        # Classification layer hyperparameters
        self.cla_param = {}
        self.cla_param["lamb"] = args.cla_lamb

        # Shared hyperparameters
        self.lr = args.lr
        self.eps = args.eps

        # Setting up layers of the network
        input_layer = InputLayer(args.train_data, args.train_label, args.train_filename, args.test_data, args.test_label, args.test_filename)
        hebbian_layer = HebbianLayer(self.input_dim, self.heb_dim, self.device_id, self.heb_param["lamb"], self.lr, self.heb_param["gam"], self.eps)
        classification_layer = ClassifierLayer(self.heb_dim, self.output_dim, self.device_id, self.cla_param["lamb"], self.lr, self.eps)
        
        self.add_module("Input Layer", input_layer)
        self.add_module("Hebbian Layer", hebbian_layer)
        self.add_module("Classification Layer", classification_layer)


    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data into the network
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the network
    """
    def forward(self, x, clamped_output=None):
        # Get all the layers in the module
        input_layer = self.get_module("Input Layer")
        hebbian_layer = self.get_module("Hebbian Layer")
        classification_layer = self.get_module("Classification Layer")

        # Convert input to float if not already
        if x.dtype != torch.float32:
            x = x.float().to(self.device_id)

        # Input data -> Hebbian Layer -> Classification Layer -> Output data
        data_input = x.to(self.device_id)
        data_input = hebbian_layer(data_input)
        data_input = classification_layer(data_input, clamped_output)

        return data_input
    
