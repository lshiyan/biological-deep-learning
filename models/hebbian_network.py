import argparse
from typing import Optional
import torch
from interfaces.network import Network
from layers.base.classification_layer import ClassificationLayer
from layers.base.data_setup_layer import DataSetupLayer
from layers.base.hebbian_layer import HebbianLayer
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer
from utils.experiment_constants import ActivationMethods, BiasUpdate, Focus, LateralInhibitions, LayerNames, LearningRules, ParamInit, WeightDecay, WeightGrowth


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
        bias_update_mapping: dict[str, BiasUpdate] = {member.value.upper(): member for member in BiasUpdate}
        focus_mapping: dict[str, Focus] = {member.value.upper(): member for member in Focus}
        activation_mapping: dict[str, ActivationMethods] = {member.value.upper(): member for member in ActivationMethods}
        
        self.heb_lamb: float = args.heb_lamb
        self.heb_eps: float = args.heb_eps
        self.heb_gam: float = args.heb_gam
        self.heb_inhib: LateralInhibitions = inhibition_mapping[args.heb_inhib.upper()]
        self.heb_learn: LearningRules = learning_rule_mapping[args.heb_learn.upper()]
        self.heb_growth: WeightGrowth = weight_growth_mapping[args.heb_growth.upper()]
        self.heb_bias_update: BiasUpdate = bias_update_mapping[args.heb_bias.upper()]
        self.heb_focus: Focus = focus_mapping[args.heb_focus.upper()]
        self.heb_act: ActivationMethods = activation_mapping[args.heb_act.upper()]

        # Classification layer hyperparameters
        self.class_learn: LearningRules = learning_rule_mapping[args.class_learn.upper()]
        self.class_growth: WeightGrowth = weight_growth_mapping[args.class_growth.upper()]
        self.class_bias_update: BiasUpdate = bias_update_mapping[args.class_bias.upper()]
        self.class_focus: Focus = focus_mapping[args.class_focus.upper()]
        self.class_act: ActivationMethods = activation_mapping[args.class_act.upper()]

        # Shared hyperparameters
        self.lr: float = args.lr
        self.sig_k: float = args.sigmoid_k
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
                                                  self.heb_inhib,
                                                  self.heb_learn,
                                                  self.heb_growth,
                                                  self.heb_bias_update,
                                                  self.heb_focus,
                                                  self.heb_act)
        classification_layer: OutputLayer = ClassificationLayer(self.heb_dim, 
                                                                self.output_dim, 
                                                                self.device, 
                                                                self.lr,
                                                                self.alpha,
                                                                self.beta,
                                                                self.sigma,
                                                                self.mu,
                                                                self.sig_k,
                                                                self.init,
                                                                self.class_learn,
                                                                self.class_growth,
                                                                self.class_bias_update,
                                                                self.class_focus
                                                                )
        
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

        # Flatten the input if necessary
        input = input.view(input.size(0), -1)  # Flatten input to (batch_size, 3*28*28)
        
        if not reconstruct:
            input = input.to(self.device)
            input = hebbian_layer(input, freeze=freeze)
            output = classification_layer(input, clamped_output) 
        elif reconstruct:
            input = input.to(self.device)
            output = hebbian_layer(input)
        
        return output
            