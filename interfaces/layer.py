from abc import ABC
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class NetworkLayer (nn.Module, ABC):
    """
    INTERFACE
    Single layer of the ANN -> Every layer of the interface must implement interface
    
    @instance attr.
        input_dimension (int): number of inputs into the layer
        output_dimension (int): number of outputs from layer
        device (str): the device that the module will be running on
        lr (float): how fast model learns at each iteration
        fc (nn.Linear): fully connected layer using linear transformation
    """
    def __init__(self, input_dimension: int, 
                 output_dimension: int, 
                 device: str,
                 learning_rate: float = 0.005) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device to which matrices will be sent
            learning_rate: how fast model learns at each iteration
        @return
            None
        """
        super ().__init__()
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.device: str = device
        self.lr: float = learning_rate
        self.fc: nn.Linear = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        
        # Setup linear activation
        for param in self.fc.parameters():
            torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)


    def create_id_tensors(self) -> torch.Tensor:   
        """
        METHOD
        Create an identity tensor
        @param
            None
        @return
            id_tensor: 3D tensor with increasing size of identify matrices
        """
        id_tensor: torch.Tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity: torch.tensor = torch.eye(i+1)
            padded_identity: torch.Tensor = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None) -> None:
        """
        METHOD
        Define the way the weights will be updated at each iteration of the training
        @param
            input: the inputs into the layer
            output: the output of the layer
            clamped_output: true labels
        @return
            None
        """
        raise NotImplementedError("This method is not implemented.")


    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        raise NotImplementedError("This method is not implemented.")
    

    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how input data flows throw the network
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            input: returns the data after passing it throw the layer
        """
        if self.training:
            input = self._train_forward(input, clamped_output)
        else:
            input = self._eval_forward(input)
        return input


    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            output: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method is not implemented.")
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            output: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method is not implemented.")


    def visualize_weights(self, result_path: str, num: int, use: str, name: str) -> None:
        """
        METHOD
        Vizualize the weight/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
            name: name to be used for folder/file
        @return
            None
        """
        # Name of saved plot
        plot_name: str = f'/{name}/{name.lower()}layerweights-{num}-{use}.png'
        
        # Find value for row and column
        row: int = 0
        col: int = 0

        root: int = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Get the weights and create heatmap
        weight: nn.parameter.Parameter = self.fc.weight
        fig: matplotlib.figure.Figure = None
        axes: np.ndarray = None
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector: torch.Tensor = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap: torch.Tensor = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
            
            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device)
        
        file_path: str = result_path + plot_name
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
    
    
    def active_weights(self, beta: float) -> int:
        """
        METHOD
        Get number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        weights: nn.parameter.Parameter = self.fc.weight
        active: torch.Tensor = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)
