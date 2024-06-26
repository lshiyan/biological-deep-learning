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
        super().__init__(args.device_id)            # SIMILARLY, I CAN DELETE THIS DEVICE ID IF THE PREVIOUS CHANGE IS IMPLEMENTED

        # Dimension of each layer
        self.input_dim = args.input_dim
        self.heb_dim = args.heb_dim
        self.output_dim = args.output_dim

        # Hebbian layer hyperparameters stored in dictionary
        self.heb_param = {
            "lamb": args.heb_lamb,
            "gam": args.heb_gam
        }

        # Classification layer hyperparameters stored in dictionary
        self.cla_param = {
            "lamb": args.cla_lamb
        }

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
        # In the above block, I register each layer with the network using add_module, 
            # note that this is a method from PyTorch’s nn.Module that allows the network to keep track of its submodules for purposes like parameter updates, backpropagation, and saving/loading models.








    """
    Method that defines how an input data flows throw the network
    @param
        x (torch.Tensor) = input data into the network
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the network
    """
    def forward(self, x, clamped_output=None):

    # STEP 1: Layer retrieval
        hebbian_layer = self.get_module("Hebbian Layer")
        classification_layer = self.get_module("Classification Layer")

    # STEP 2: Input type conversion
        if x.dtype != torch.float32:
            x = x.float().to(self.device_id)
        # This step is crucial for ensuring compatibility with PyTorch operations, which often require floating-point precision.

    # STEP 3: DATA flow through layers
        data_input = x.to(self.device_id)                                                       # The input data is first moved to the correct device, if not already done in the type conversion step.
        post_hebbian_value = hebbian_layer(data_input)                                          # The data then flows through the Hebbian Layer. 
        post_classification_value = classification_layer(post_hebbian_value, clamped_output)    # Lastly, the data the processed data is passed to the Classification Layer. 
                                                                                                    # If clamped_output is provided, it is used during this stage. 

        return post_classification_value
    
