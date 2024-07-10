import argparse
from enum import Enum
from typing import Optional, Union
import torch
from interfaces.network import Network
from layers.base.classification_layer import ClassificationLayer
from layers.base.data_setup_layer import DataSetupLayer
from layers.base.hebbian_layer import HebbianLayer
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer
from utils.experiment_constants import LateralInhibitions, LayerNames, LearningRules, WeightGrowth


class HebbianNetwork(Network):
    """
    CLASS
    Defines the base hebbian network
    @instance attr.
        PARENT ATTR.
            device (int) = id of the gpu that the model will be running in
        OWN ATTR.
            input_dim (int): number of inputs
            heb_dim (int): number of neurons in hebbina layer
            output_dimension (int): number of output neurons
            heb_param (dict {str:float}): dictionary with all the hyperparameters for the hebbian layer
                - lr (float): learning rate of hebbian layer
                - lamb (float): hyperparameter for lateral neuron inhibition
                - gam (float): factor to decay learning rate of hebbian layer
                - eps (float): small value to avoid 0 division
            cla_param (dict {str:float}): dictionary with all the hyperparameters for the classification layer
                - in_1 (bool): wether to include first neuron in classification layer
            lr (float): learning rate of classification layer
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
        inhibition_mapping: dict[str, LateralInhibitions] = {member.value.upper(): member for member in LateralInhibitions}
        learning_rule_mapping: dict[str, LearningRules] = {member.value.upper(): member for member in LearningRules}
        weight_growth_mapping: dict[str, WeightGrowth] = {member.value.upper(): member for member in WeightGrowth}
        
        self.heb_param: dict[str, Union[float, Enum]] = {
            "lamb": args.heb_lamb,
            "eps": args.heb_eps,
            "gam": args.heb_gam,
            "inhib": inhibition_mapping[args.inhibition_rule.upper()],
            "learn": learning_rule_mapping[args.learning_rule.upper()],
            "growth": weight_growth_mapping[args.weight_growth.upper()],
            "sig_k": args.sigmoid_k
        }

        # Classification layer hyperparameters stored in dictionary
        self.cla_param: dict[str, Union[float, Enum]] = {
            "in_1": args.include_first
        }

        # Shared hyperparameters
        self.lr: float = args.lr

        # Setting up layers of the network
        input_layer: InputLayer = DataSetupLayer()
        hebbian_layer: HiddenLayer = HebbianLayer(self.input_dim, 
                                                  self.heb_dim, 
                                                  self.device, 
                                                  self.heb_param["lamb"],  # type: ignore
                                                  self.lr, 
                                                  self.heb_param["gam"],  # type: ignore
                                                  self.heb_param["eps"], # type: ignore
                                                  self.heb_param["sig_k"], # type: ignore
                                                  self.heb_param["inhib"], # type: ignore
                                                  self.heb_param["learn"], # type: ignore
                                                  self.heb_param["growth"]) # type: ignore
        classification_layer: OutputLayer = ClassificationLayer(self.heb_dim, 
                                                                self.output_dim, 
                                                                self.device, 
                                                                self.lr,
                                                                self.cla_param["in_1"]) # type: ignore
        
        self.add_module(LayerNames.INPUT.name, input_layer)
        self.add_module(LayerNames.HIDDEN.name, hebbian_layer)
        self.add_module(LayerNames.OUTPUT.name, classification_layer)


    def forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None, reconstruct: bool = False) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network
        @param
            input: input data into the network
            clamped_output: one-hot encode of true labels
            reconstruct: wether to use reconstruction for this model
            freeze: wether to freeze the hebbian layer during training
        @return
            output: returns the data after passing it throw the network
        """
        # Get layers of network
        hebbian_layer = self.get_module(LayerNames.HIDDEN)
        classification_layer = self.get_module(LayerNames.OUTPUT)

        #if self.training and not reconstruct:
        #    print(f"hebbian_network -> freeze: {freeze}")

        if not reconstruct:
            input = input.to(self.device)
            # print(f"currently freeze is: {freeze}")
            input = hebbian_layer(input)
            output = classification_layer(input, clamped_output) 
        elif reconstruct:
            input = input.to(self.device)
            output = hebbian_layer(input)
        
        return output
            