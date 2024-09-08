import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
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


from abc import ABC
import math
from typing import Optional, Tuple
import matplotlib
import matplotlib.figure
import matplotlib.colors as mcolors
import os

from utils.experiment_constants import LayerNames, ParamInit
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Optional
from layers.base.data_setup_layer import DataSetupLayer

from interfaces.network import Network
from layers.base.classification_layer import ClassificationLayer
from layers.base.data_setup_layer import DataSetupLayer
from layers.base.hebbian_layer import HebbianLayer
from layers.hidden_layer import HiddenLayer
from layers.input_layer import InputLayer
from layers.output_layer import OutputLayer
from utils.experiment_constants import ActivationMethods, BiasUpdate, Focus, LateralInhibitions, LayerNames, LearningRules, ParamInit, WeightDecay, WeightGrowth

# Import growth functions from the utils.weight_growth_fcts module
from utils.weight_growth_fcts import linear_growth, sigmoid_growth, exponential_growth, neuron_norm

class SGDNetwork(nn.Module):
    def __init__(self, name: str, args: argparse.Namespace) -> None:
        super(SGDNetwork, self).__init__()

        # Set dimensions and hyperparameters
        self.input_dim = args.input_dim
        self.heb_dim = args.heb_dim
        self.output_dim = args.output_dim
        self.lr = args.lr
        self.sig_k = args.sigmoid_k
        self.heb_focus = Focus[args.heb_focus.upper()]  # Assuming Focus is Enum
        self.heb_growth = WeightGrowth[args.heb_growth.upper()]  # Assuming WeightGrowth is Enum
        self.heb_lamb: float = args.heb_lamb
        self.beta: float = args.beta

        # Select the derivative function based on heb_growth
        if self.heb_growth == WeightGrowth.LINEAR:
            self.derivative = linear_growth
        elif self.heb_growth == WeightGrowth.SIGMOID:
            self.derivative = partial(sigmoid_growth, plasticity=self.heb_focus, k=self.sig_k)
        elif self.heb_growth == WeightGrowth.EXPONENTIAL:
            self.derivative = partial(exponential_growth, plasticity=self.heb_focus)
        else:
            raise ValueError(f"Growth type {self.heb_growth} not supported.")

        # Set up layers without biases
        input_layer = DataSetupLayer()  # Example input layer
        self.hidden_layer = nn.Linear(self.input_dim, self.heb_dim, bias=False)  # No bias
        self.output_layer = nn.Linear(self.heb_dim, self.output_dim, bias=False)  # No bias

        # Activation function
        self.relu = nn.ReLU()

        # Optimizer and loss function
        #self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()  # Handles the softmax internally, expects logits and class indices

        # Register layers
        self.add_module(input_layer.name.name, input_layer)
        self.add_module('HIDDEN', self.hidden_layer)
        self.add_module('OUTPUT', self.output_layer)
        self.init_weights_with_beta()


    def init_weights_with_beta(self):

        for name, param in self.named_parameters():
            if self.heb_focus == Focus.NEURON:
                if param.data.ndim == 1:
                    param.data = self.beta * param.data
                elif param.data.ndim == 2:
                    param.data = self.beta * param.data / neuron_norm(param.data, self.sig_k)
                else:
                    raise NotImplementedError("Weight inits only implemented for rank 1 and 2 tensors.")

            elif self.heb_focus == Focus.SYNASPSE:
                param.data = self.beta * param.data
            else:
                raise ValueError("Illegal focus.")


    def new_weight(self, old_w, grad):
        """Updates the weights using the derivative function and the provided gradient."""
        if grad is None:
            print(f"Skipping weight update for {old_w} because grad is None.")
            return old_w

        # Apply the chosen derivative function to all weights
        derivative_value = self.derivative(old_w)
        print(f"Derivative value mean: {derivative_value.mean().item()}")  # Debug statement
        new_weight = old_w - self.lr * derivative_value * grad  # Gradient descent

        return new_weight

    def update_weights(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        """Updates weights based on the loss between logits and target."""
        print(f"Logits shape: {logits.shape}, Target shape: {target.shape}")
        assert logits.size(0) == target.size(0), \
            f"Logits batch size {logits.size(0)} does not match target batch size {target.size(0)}"

        # Compute loss and backpropagate
        #self.optimizer.zero_grad()
        loss = self.loss_fn(logits, target)
        loss.backward()

        # Manually update weights using the custom derivative
        for name, param in self.named_parameters():
            if param.grad is not None:
                param.data = self.new_weight(param.data, param.grad)
            else:
                print(f"Warning: Gradient for parameter {param} is None. Skipping update.")

        return loss.item()

    def forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward method for processing input through the network."""
        input = input.view(input.size(0), -1)  # Flatten input
        h = self.relu(self.hidden_layer(input))
        logits = self.output_layer(h)

        if clamped_output is not None:
            self.update_weights(logits, clamped_output)

        return logits
    

    def get_module(self, lname: LayerNames) -> nn.Module:
        """
        METHOD
        Returns layer with given name
        @param
            name: name of layer to get
        @return
            layer: layer of the network with searched name
        """
        for layer_name, layer in self.named_children():      
            if lname.name.upper() == layer_name.upper():
                return layer
        raise NameError(f"There are no layer named {lname.name.upper()}.")
    
    def visualize_weights(self, result_path: str, epoch: int, use: str) -> None:
        """
        Visualizes the weights/features learned by neurons in the hidden and output layers using heatmaps.
        This function automatically detects and visualizes layers of interest.
        @param
            result_path: path to folder where results will be printed
            epoch: training epoch current training loop is at (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
        @return
            None
        """
        # Loop through relevant layers
        for layer_name in ['HIDDEN', 'OUTPUT']:
            # Get the layer by name
            try:
                layer = self.get_module(LayerNames[layer_name])
            except NameError as e:
                print(f"Skipping layer {layer_name}: {e}")
                continue
            
            if not isinstance(layer, nn.Linear):
                print(f"Skipping layer {layer_name}: Not a Linear layer.")
                continue

            # Retrieve the weights
            weight = layer.weight.data  # Get the weights tensor of the layer

            # Name of saved plot
            plot_name = f'/{use}/{use.lower()}_{layer_name.lower()}_weights-{epoch}.png'
            
            # Determine number of rows and columns for plotting
            output_dim = weight.size(0)
            row, col = self.row_col(output_dim)

            # Calculate size of figure
            subplot_size = 4
            fig_width = col * subplot_size
            fig_height = row * subplot_size

            # Create the heatmap plots
            fig, axes = plt.subplots(row, col, figsize=(fig_width, fig_height))
            for ele in range(row * col):
                if ele < output_dim:
                    # Move tensor to CPU and convert to NumPy array for visualization
                    weights = weight[ele].cpu().numpy().reshape(-1, 1)
                    feature_row, feature_col = self.row_col(weights.size)

                    # Define the colormap based on weight values
                    max_value = np.max(weights)
                    min_value = np.min(weights)
                    cmap = self.get_cmap(min_value, max_value)

                    ax = axes[ele // col, ele % col]
                    im = ax.imshow(weights.reshape(feature_row, feature_col), cmap=cmap, interpolation='nearest', vmin=min_value, vmax=max_value)
                    cbar = fig.colorbar(im, ax=ax)
                    ax.set_title(f'Neuron {ele} Weights')

                    # Set color bar ticks
                    ticks = np.linspace(min_value, max_value, num=5).tolist()
                    ticks.append(0)
                    ticks = sorted(set(ticks))
                    cbar.set_ticks(ticks)
                else:
                    axes[ele // col, ele % col].axis('off')

            # Save the plot
            file_path = result_path + plot_name
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()

    def row_col(self, num_elements: int) -> tuple:
        """
        Computes the number of rows and columns for a subplot layout given the number of elements.
        @param
            num_elements: the total number of elements to display in subplots
        @return
            A tuple of (rows, columns)
        """
        row = int(np.ceil(np.sqrt(num_elements)))
        col = int(np.ceil(num_elements / row))
        return row, col

    def get_cmap(self, min_value: float, max_value: float):
        """
        Generates a colormap scaled to the range of the weight values.
        @param
            min_value: minimum weight value
            max_value: maximum weight value
        @return
            Colormap
        """
        return plt.get_cmap('coolwarm')

    def active_weights(self, beta: float) -> dict[str, int]:
        """
        METHOD
        Returns number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            module_active_weights: dictionary {str:int}
        """
        module_active_weights: dict[str, int] = {}
        
        for name, module in self.name_children():
            module_active_weights[name.lower()] = module.active_weights(beta)
        
        return module_active_weights