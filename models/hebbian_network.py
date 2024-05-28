import torch.nn as nn
from layers.input_layer import InputLayer
from layers.hebbian_layer import HebbianLayer
from layers.classifier_layer import ClassifierLayer
from models.network import Network 

class HebbianNetwork(Network):
    """
    Constructor method
    @attr.
        PARENT ATTR.
            __layers (dict {str:layers.NetworkLayer}) = list of layers of the network
                input_layer (layers.InputLayer) = layer that will take care of input processing
                hebbian_layer (layers.HebbianLayer) = layer for hebbian learning
                output_layer (layers.ClassifierLayer) = layer for classification task
        OWN ATTR.
            input_dim (int) = number of inputs
            heb_dim (int) = number of neurons in hebbina layer
            output_dimension (int) = number of output neurons
            heb_param (dict {str:float}) = dictionary with all the hyperparameters for the hebbian layer
                lr (float) = learning rate of hebbian layer
                lamb (float) = hyperparameter for lateral neuron inhibition
                gam (float) = factor to decay learning rate of hebbian layer
            cla_param (dict {str:float}) = dictionary with all the hyperparameters for the classification layer
                lr (float) = learning rate of classification layer
                lamb (float) = hyperparameter for lateral neuron inhibition
                gam (float) = factor to decay learning rate of classification layer
            eps (float) = small value to avoid 0 division
    @pram
        args (argparse.ArgumentParser) = arguments passed from command line
    """
    def __init__(self, args):
        super().__init__()

        # Dimension of each layer
        self.input_dim = args.input_dim
        self.heb_dim = args.heb_dim
        self.output_dim = args.output_dim

        # Hebbian layer hyperparameters
        self.heb_param = {}
        self.heb_param["lr"] = args.heb_lr
        self.heb_param["lamb"] = args.heb_lamb
        self.heb_param["gam"] = args.heb_gam

        # Classification layer hyperparameters
        self.cla_param = {}
        self.cla_param["lr"] = args.cla_lr
        self.cla_param["lamb"] = args.cla_lamb
        self.cla_param["gam"] = args.cla_gam

        # Shared hyperparameters
        self.eps = args.eps

        # Setting up layers of the network
        input_layer = InputLayer(args.train_data, args.train_label, args.train_filename, args.test_data, args.test_label, args.test_filename)
        hebbian_layer = HebbianLayer(self.input_dim, self.heb_dim, self.heb_param["lamb"], self.heb_param["lr"], self.heb_param["gam"], self.eps)
        classification_layer = ClassifierLayer(self.heb_dim, self.output_dim, self.cla_param["lamb"], self.cla_param["lr"], self.cla_param["gam"], self.eps)
        self.add_layer("Input Layer", input_layer)
        self.add_layer("Hebbian layer", hebbian_layer)
        self.add_layer("Classification Layer", classification_layer)


    """
    Method that defines how an input data flows throw the network
    @param
        data_input (torch.Tensor) = input data into the network
        clamped_output (???) = parameter to clamp the output   # WTV this means
    """   
    def forward(self, data_input, clamped_output=None):
        for name, module in self.named_children():
            if name == 'Hebbian Layer':
                data_input = module(data_input, clamped_output)
            elif name == 'Classification Lyaer':
                data_input = module(data_input)
        return data_input