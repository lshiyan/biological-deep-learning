# Built-in imports
import time
from typing import Tuple, Type

# Pytorch imports
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from torch import linalg as LA
import torch.nn as nn

# Custom defined model imports
from interfaces.experiment import Experiment
from interfaces.network import Network
from layers.input_layer import InputLayer

from layers.hopfield_sanger.hsang_input_layer import HSangInputLayer
from layers.hopfield_sigmoid.hsig_input_layer import HSigInputLayer
from layers.hopfield_YZZ.hyzz_input_layer import HYZZInputLayer

from layers.relu_sanger.rsang_input_layer import RSangInputLayer
from layers.relu_sigmoid.rsig_input_layer import RSigInputLayer
from layers.relu_YZZ.ryzz_input_layer import RYZZInputLayer

from layers.softmax_sanger.ssang_input_layer import SSangInputLayer
from layers.softmax_sigmoid.ssig_input_layer import SSigInputLayer
from layers.softmax_YZZ.syzz_input_layer import SYZZInputLayer

from layers.extended_hebbian.extended_input_layer import EHebInputLayer
from layers.extended_hebbian.extended_hebbian_layer import EHebHebbianLayer
from layers.extended_hebbian.extended_classification_layer import EHebClassificationLayer

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



class EHebExperiment(Experiment):

    """
    Class used for out-of-distribution testing
    
    """

    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:
        """
        CONSTRUCTOR METHOD
        
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
        """
        super().__init__(model, args, name)

    
    def compute_reconstruction_norm_difference(self, data_loader: DataLoader) -> float:

        # First, I set the model to evaluation mode
        self.model.eval()

        # Now, I initialize the total error to be zero
        total_error = 0.0

        # Now, I disable the gradient calculation
        with torch.no_grad():

            # Loop through each batch in data loader

            for batch in data_loader:

                input, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Calcuate the L2 norm, this is to normalize the data
                norm_reconstructed = LA.vector_norm(reconstructed_input, ord=2)
                norm_input = LA.vector_norm(inputs, ord=2)


                # Calculate the norm difference as specified by prof
                    # This should mimic: (|| (x_tilt / ||x_tilt|| ) - (x / ||x||) || )^2

                curr_error = LA.vector_norm( (reconstructed_input / norm_reconstructed)  -  (inputs / norm_input)  ) 

                curr_error_squared = torch.pow(curr_error, exp=2)
        
                # Increment total error
                total_error += curr_error_squared 
            
            # Now, I return the average reconstruction norm difference error
            return total_error / len(data_loader.dataset)

            
    
    
    def compute_reconstruction_cosine_difference(self, data_loader: DataLoader) -> float:

        # First, I set the model to evaluation mode
        self.model.eval()

        # Now, I initialize the total error to be zero
        total_error = 0.0

        # Define the Cosine Similarity operation
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Now, I disable the gradient calculation
        with torch.no_grad():

            # Loop through each batch in data loader

            for batch in data_loader:

                input, _ = batch

                inputs = inputs.to(self.ARGS.device).float()    # Move input to device 

                # The next step is to reconstruct the input from the hidden activation using the Hebbian Layer weights
                hidden_activations = self.model.get_module("Hebbian")(inputs)

                # Getting transpose weight
                W_transpose = self.model.get_module("Hebbian").fc.weights.T

                # reconstructed input using hidden activations
                reconstructed_input = torch.matmul(hidden_activations, W_transpose)

                # Compute the cosine similarity
                cos_error = cos(inputs, reconstructed_input)

                # Convert to a scalar by taking the mean
                scalar_cos_error = cos_error.mean()

                # Increment total error
                total_error += scalar_cos_error 
        
        
            # Now, I return the average reconstruction norm difference error
            return total_error / len(data_loader.dataset)
































































































