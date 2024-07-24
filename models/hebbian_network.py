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
from utils.experiment_constants import BiasUpdate, LateralInhibitions, LayerNames, LearningRules, ParamInit, WeightDecay, WeightGrowth


class HebbianNetwork(Network):
    """
    CLASS
    Defines the base hebbian network
    @instance attr.
        PARENT ATTR.
            name (str): name of network
            device (int) = id of the gpu that the model will be running in
        OWN ATTR.
            input_dim (int): number of inputs
            heb_dim (int): number of neurons in hebbina layer
            output_dimension (int): number of output neurons
            lamb (float): hyperparameter for lateral neuron inhibition
            gam (float): factor to decay learning rate of hebbian layer
            eps (float): small value to avoid 0 division
            inhib (LateralInhibition): type of lateral inhibition
            learn (LearningRules): which learning rule that will be used
            growth (WeightGrowth): type of weight growth
            sig_k (float): sigmoid learning constant
            in_1 (bool): wether to include first neuron in classification layer
            lr (float): learning rate of classification layer
            alpha (flaot): lower bound for random uniform
            beta (flaot): upper bound for random uniform
            sigma (flaot): variance for random normal
            mu (flaot): mean for random normal
            init (ParamInit): type of parameter initiation
    """
    def __init__(self, name, args: argparse.Namespace) -> None:
        """
        CONSTRUCTOR METHOD
        @pram
            args: arguments passed from command line
        @return
            None
        """
        super().__init__(name, args.device)

        # Dimension of each layer
        self.input_dim: int = args.input_dim
        self.heb_dim: int = args.heb_dim
        self.output_dim: int = args.output_dim

        # Hebbian layer hyperparameters
        inhibition_mapping: dict[str, LateralInhibitions] = {member.value.upper(): member for member in LateralInhibitions}
        learning_rule_mapping: dict[str, LearningRules] = {member.value.upper(): member for member in LearningRules}
        weight_growth_mapping: dict[str, WeightGrowth] = {member.value.upper(): member for member in WeightGrowth}
        param_init_mapping: dict[str, ParamInit] = {member.value.upper(): member for member in ParamInit}
        weight_decay_mapping: dict[str, WeightDecay] = {member.value.upper(): member for member in WeightDecay}
        bias_update_mapping: dict[str, BiasUpdate] = {member.value.upper(): member for member in BiasUpdate}
        
        self.heb_lamb: float = args.heb_lamb
        self.heb_eps: float = args.heb_eps
        self.heb_gam: float = args.heb_gam
        self.sig_k: float = args.sigmoid_k
        self.inhib: LateralInhibitions = inhibition_mapping[args.inhibition_rule.upper()]
        self.learn: LearningRules = learning_rule_mapping[args.learning_rule.upper()]
        self.growth: WeightGrowth = weight_growth_mapping[args.weight_growth.upper()]
        self.weight_decay: WeightDecay = weight_decay_mapping[args.weight_decay.upper()]
        self.bias_update: BiasUpdate = bias_update_mapping[args.bias_update.upper()]

        # Classification layer hyperparameters
        self.include_first = args.include_first

        # Shared hyperparameters
        self.lr: float = args.lr
        self.alpha: float = args.alpha
        self.beta: float = args.beta
        self.sigma: float = args.sigma
        self.mu: float = args.mu
        self.init: ParamInit = param_init_mapping[args.init.upper()]

        # Setting up layers of the network
        input_layer: InputLayer = DataSetupLayer()
        hebbian_layer: HiddenLayer = HebbianLayer(self.input_dim, 
                                                  self.heb_dim, 
                                                  self.device, 
                                                  self.heb_lamb,
                                                  self.lr,
                                                  self.alpha,
                                                  self.beta,
                                                  self.sigma,
                                                  self.mu,
                                                  self.init, 
                                                  self.heb_gam,
                                                  self.heb_eps,
                                                  self.sig_k,
                                                  self.inhib,
                                                  self.learn,
                                                  self.growth,
                                                  self.weight_decay,
                                                  self.bias_update)
        classification_layer: OutputLayer = ClassificationLayer(self.heb_dim, 
                                                                self.output_dim, 
                                                                self.device, 
                                                                self.lr,
                                                                self.alpha,
                                                                self.beta,
                                                                self.sigma,
                                                                self.mu,
                                                                self.init,
                                                                self.include_first)
        
        self.add_module(input_layer.name.name, input_layer)
        self.add_module(hebbian_layer.name.name, hebbian_layer)
        self.add_module(classification_layer.name.name, classification_layer)


    def forward(self, 
                input: torch.Tensor, 
                clamped_output: Optional[torch.Tensor] = None, 
                reconstruct: bool = False, 
                freeze: bool = False
                ) -> torch.Tensor:
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

        if not reconstruct:
            input = input.to(self.device)
            input = hebbian_layer(input, freeze=freeze)
            output = classification_layer(input, clamped_output) 
        elif reconstruct:
            input = input.to(self.device)
            output = hebbian_layer(input)
        
        return output
            