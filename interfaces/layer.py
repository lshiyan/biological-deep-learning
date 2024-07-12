from abc import ABC
import math
from typing import Optional, Tuple
import matplotlib
import matplotlib.figure
from networkx import reconstruct_path

from utils.experiment_constants import LayerNames, ParamInit
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class NetworkLayer (nn.Module, ABC):
    """
    INTERFACE
    Defines a single layer of an ANN -> Every layer of the interface must implement interface
    @instance attr.
        input_dimension (int): number of inputs into the layer
        output_dimension (int): number of outputs from the layer
        device (str): device that will be used for CUDA
        lr (float): how fast model learns at each iteration
        fc (nn.Linear): fully connected layer using linear transformation
        alpha (float): lower bound for random unifrom 
        beta (float): upper bound for random uniform
        sigma (float): variance for random normal
        mu (float): mean for random normal
        init (ParamInit): fc parameter initiation type
    """
    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 device: str = 'cpu',
                 learning_rate: float = 0.005,
                 alpha: float = 0,
                 beta: float = 1,
                 sigma: float = 1,
                 mu: float = 0,
                 init: ParamInit = ParamInit.UNIFORM
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
            alpha (float): lower bound for random unifrom 
            beta (float): upper bound for random uniform
            sigma (float): variance for random normal
            mu (float): mean for random normal
            init (ParamInit): fc parameter initiation type
        @return
            None
        """
        super ().__init__()
        self.name: LayerNames
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.device: str = device
        self.lr: float = learning_rate
        self.fc: nn.Linear = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        nn.init.zeros_(self.fc.bias)
        self.alpha: float = alpha
        self.beta: float = beta
        self.sigma: float = sigma
        self.mu: float = mu
        self.init: ParamInit = init
        
        # Initialize fc layer parameters
        for param in self.fc.parameters():
            if self.init == ParamInit.UNIFORM:
                torch.nn.init.uniform_(param, a=alpha, b=beta)
            elif self.init == ParamInit.NORMAL:
                torch.nn.init.normal_(param, mean=mu, std=sigma)
            else:
                raise NameError(f"Invalid parameter init {self.init}.")
            
            param.requires_grad_(False)

        # Frozen or not for the layer
        self.freeze = False


    def create_id_tensors(self) -> torch.Tensor:   
        """
        METHOD
        Creates an identity tensor
        @param
            None
        @return
            id_tensor: 3D tensor with increasing size of identify matrices
        """
        id_tensor: torch.Tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity: torch.Tensor = torch.eye(i+1)
            padded_identity: torch.Tensor = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError("This method is not implemented.")


    def update_bias(self, output: torch.Tensor) -> None:
        raise NotImplementedError("This method is not implemented.")
    

    def forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None, freeze: bool = False) -> torch.Tensor:
        """
        METHOD
        Defines how input data flows through the network
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
            freeze: determine if layer is frozen or not
        @return
            output: returns the data after passing it through the layer
        """
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        output: torch.Tensor
        if self.training and not freeze:
            output = self._train_forward(input_copy, clamped_output)
        else:
            output = self._eval_forward(input_copy)
        return output


    def _train_forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented.")
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented.")


    def visualize_weights(self, result_path: str, num: int, use: str, fname: str) -> None:
        """
        METHOD
        Vizualizes the weights/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
            fname: name to be used for folder/file
        @return
            None
        """
        # Name of saved plot
        plot_name: str = f'/{fname}/{fname.lower()}layerweights-{num}-{use}.png'
        
        # Find value for row and column
        row: int
        col: int
        row, col = self.row_col(self.output_dimension)
        
        # Calculate size of figure
        subplot_size = 3  # You can adjust this value to make subplots larger or smaller
        fig_width = col * subplot_size
        fig_height = row * subplot_size

        # Get the weights and create heatmap
        weight: torch.Tensor = self.fc.weight
        fig: matplotlib.figure.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(row, col, figsize=(fig_width, fig_height)) # type: ignore
        for ele in range(row * col): 
            if ele < self.output_dimension:
                random_feature_selector: torch.Tensor = weight[ele]
                # Move tensor to CPU, convert to NumPy array for visualization
                feature_row, feature_col = self.row_col(random_feature_selector.size(0))
                original_size: int = random_feature_selector.size(0)
                plot_size: int = feature_row * feature_col
                padding_size: int = plot_size - original_size
                padded_weights: torch.Tensor = torch.nn.functional.pad(random_feature_selector, (0, padding_size))
                heatmap: torch.Tensor = padded_weights.view(feature_row, feature_col).cpu().numpy()
                ax = axes[ele // col, ele % col]
                im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
                fig.colorbar(im, ax=ax)
                ax.set_title(f'Weight {ele}')
                
                # Move the tensor back to the GPU if needed
                random_feature_selector = random_feature_selector.to(self.device)
            else:
                ax = axes[ele // col, ele % col]
                ax.axis('off')
        
        file_path: str = result_path + plot_name
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
    
    
    def active_weights(self, beta: float) -> int:
        """
        METHOD
        Returns number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        weights: torch.Tensor = self.fc.weight
        active: torch.Tensor = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)


    @staticmethod
    def row_col(num: int) -> Tuple[int, int]:
        # Find value for row and column
        row: int = 0
        col: int = 0
        
        root: int = int(math.sqrt(num))
        min_product: int = num ** 2
        
        for i in range(2, root + 1):
            if num % i == 0:
                factor1 = i
                factor2 = num // i
                
                if factor1 * factor2 >= num and factor1 * factor2 <= min_product:
                    min_product = factor1 * factor2
                    row = min(factor1, factor2)
                    col = max(factor1, factor2)
            else:
                factor1 = i
                factor2 = math.ceil(num / i)
                
                if factor1 * factor2 >= num and factor1 * factor2 <= min_product:
                    min_product = factor1 * factor2
                    row = min(factor1, factor2)
                    col = max(factor1, factor2)
        
        
        return row, col