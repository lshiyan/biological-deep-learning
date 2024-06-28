from abc import ABC
import torch
from interfaces.layer import NetworkLayer


class OutputLayer(NetworkLayer, ABC):
    """
    INTERFACE
    Output layer in ANN -> All output layers should implement this class
    
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): the device that the module will be running on
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
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
        super().__init__(input_dimension, output_dimension, device, learning_rate)
    

    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None) -> None:
        """
        METHOD
        Defines the way the weights will be updated at each iteration of the training.
        @param
            input: The input tensor to the layer before any transformation.
            output: The output tensor of the layer before applying softmax.
            clamped_output: one-hot encode of true labels
        @return
            None
        """
        raise NotImplementedError("This method has yet to be implemented.")
        

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        raise NotImplementedError("This method has yet to be implemented.")
    

    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            input: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method has yet to be implemented.")
    
    
    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            input: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method has yet to be implemented.")