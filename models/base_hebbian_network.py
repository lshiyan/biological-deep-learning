import argparse
import torch
from layers.base.base_input_layer import BaseInputLayer
from layers.base.base_hebbian_layer import BaseHebbianLayer
from layers.base.base_classification_layer import BaseClassificationLayer
from interfaces.network import Network
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer 


class BaseHebbianNetwork(Network):
    """
    CLASS
    Defining the base hebbian network
    @instance attr.
        PARENT ATTR.
            device (int) = id of the gpu that the model will be running in
        OWN ATTR.
            input_dim (int) = number of inputs
            heb_dim (int) = number of neurons in hebbina layer
            output_dimension (int) = number of output neurons
            heb_param (dict {str:float}) = dictionary with all the hyperparameters for the hebbian layer
                - lr (float) = learning rate of hebbian layer
                - lamb (float) = hyperparameter for lateral neuron inhibition
                - gam (float) = factor to decay learning rate of hebbian layer
                - eps (float) = small value to avoid 0 division
            cla_param (dict {str:float}) = dictionary with all the hyperparameters for the classification layer
            lr (float) = learning rate of classification layer
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        CONSTRUCTOR METHOD
        @pram
            args: arguments passed from command line
        @return
            None
        """
        super().__init__(args.device)

        # Dimension of each layer
        self.input_dim: int = args.input_dim
        self.heb_dim: int = args.heb_dim
        self.output_dim: int = args.output_dim

        # Hebbian layer hyperparameters stored in dictionary
        self.heb_param: dict[str:float] = {
            "lamb": args.heb_lamb,
            "eps": args.heb_eps,
            "gam": args.heb_gam
        }

        # Classification layer hyperparameters stored in dictionary
        self.cla_param: dict[:] = {}

        # Shared hyperparameters
        self.lr: float = args.lr

        # Setting up layers of the network
        input_layer: InputLayer = BaseInputLayer(args.train_data, args.train_label, args.train_filename, args.test_data, args.test_label, args.test_filename)
        hebbian_layer: HiddenLayer = BaseHebbianLayer(self.input_dim, self.heb_dim, self.device, self.heb_param["lamb"], self.lr, self.heb_param["gam"], self.heb_param["eps"])
        classification_layer: OutputLayer = BaseClassificationLayer(self.heb_dim, self.output_dim, self.device, self.lr)
        
        self.add_module("Input", input_layer)
        self.add_module("Hebbian", hebbian_layer)
        self.add_module("Classification", classification_layer)


    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network
        @param
            input: input data into the network
            clamped_output: one-hot encode of true labels
        @return
            output: returns the data after passing it throw the network
        """
        # Get layers of network
        hebbian_layer = self.get_module("Hebbian")
        classification_layer = self.get_module("Classification")

        # Feedforward data input into network
        input = input.to(self.device)
        input = hebbian_layer(input)
        output = classification_layer(input, clamped_output)

        return output