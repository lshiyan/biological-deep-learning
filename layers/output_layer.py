from abc import ABC
from typing import Optional
import torch
from interfaces.layer import NetworkLayer


class OutputLayer(NetworkLayer, ABC):
    """
    INTERFACE
    Defines the output layer of an ANN -> Every output layer should implement this class
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): device that will be used for CUDA
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
    """
    def __init__(self, 
                 input_dimension: int,
                 output_dimension: int, 
                 device: str, 
                 learning_rate: float = 0.005
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device, learning_rate)
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> None:
        raise NotImplementedError("This method has yet to be implemented.")
        

    def update_bias(self, output: torch.Tensor) -> None:
        raise NotImplementedError("This method has yet to be implemented.")
    

    def _train_forward(self, input: torch.Tensor, clamped_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("This method has yet to be implemented.")
    
    
    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method has yet to be implemented.")