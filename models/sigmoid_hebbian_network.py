import argparse
import torch

from layers.sigmoid.sigmoid_input_layer import SigmoidInputLayer
from layers.sigmoid.sigmoid_hebbian_layer import SigmoidHebbianLayer
from layers.sigmoid.sigmoid_classification_layer import SigmoidClassificationLayer

from interfaces.network import Network
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer 



class SigmoidHebbianNetwork(Network):
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
        input_layer: InputLayer = SigmoidInputLayer()
        hebbian_layer: HiddenLayer = SigmoidHebbianLayer(self.input_dim, self.heb_dim, self.device, self.heb_param["lamb"], self.lr, self.heb_param["gam"], self.heb_param["eps"])
        classification_layer: OutputLayer = SigmoidClassificationLayer(self.heb_dim, self.output_dim, self.device, self.lr)
        
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
        hebbian_layer = self.get_module("Hebbian")
        classification_layer = self.get_module("Classification")

        if input.dtype != torch.float32:
            input = input.float().to(self.device)

        data_input = input.to(self.device)
        post_hebbian_value = hebbian_layer(data_input)
        output = classification_layer(post_hebbian_value, clamped_output)

        return output