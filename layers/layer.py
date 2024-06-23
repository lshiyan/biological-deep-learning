from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class NetworkLayer (nn.Module, ABC):
    """
    Abstract class for a single layer of the ANN -> Every layer of the interface must implement interface
    This will help with the support of multiple hidden layers inside the network
    
    @instance attr.
        input_dimension (int): number of inputs into the layer
        output_dimension (int): number of outputs from layer
        device_id (str): the device that the module will be running on
        lamb (float): lambda hyperparameter for latteral inhibition
        alpha (float): how fast model learns at each iteration
        fc (nn.Linear): function to apply linear transformation to incoming data
        eps (float): to avoid division by 0
    """
    def __init__(self, input_dimension: int, output_dimension: int, device_id: str, lamb: float = 1, learning_rate: float = 0.005, eps: float = 0.01) -> None:
        """
        Constructor method NetworkLayer
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device_id: device to which matrices will be sent
            lamb: lambda hyperparameter for latteral inhibition
            learning_rate: how fast model learns at each iteration
            eps: to avoid division by 0
        @return
            None
        """
        super ().__init__()
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.device_id: str = device_id
        self.lamb: float = lamb
        self.alpha: float = learning_rate
        self.fc: nn.Linear = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
        self.eps: float = eps
        
        # Setup linear activation
        for param in self.fc.parameters():
            torch.nn.init.uniform_(param, a=0.0, b=1.0)
            param.requires_grad_(False)


    def create_id_tensors(self) -> torch.Tensor:   
        """
        Method to create an identity tensor
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


    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        Method that defines how an input data flows throw the network
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
    

    @abstractmethod
    def visualize_weights(self, result_path: str, num: int, use: str) -> None:
        """
        Method to vizualize the weight/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
        @return
            None
        """
        pass
    

    @abstractmethod
    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None) -> None:
        """
        Method to define the way the weights will be updated at each iteration of the training
        @param
            input: the inputs into the layer
            output: the output of the layer
            clamped_output: true labels
        @return
            None
        """
        pass


    @abstractmethod 
    def update_bias(self, output: torch.Tensor) -> None:
        """
        Method to define the way the bias will be updated at each iteration of the training
        @param
            output: the output of the layer
        @return
            None
        """
        pass
    

    @abstractmethod
    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        Method that defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: one-hot encode of true labels
        @return
            input: returns the data after passing it throw the layer
        """
        pass
    

    @abstractmethod
    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Method that defines how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            input: returns the data after passing it throw the layer
        """
        pass


    @abstractmethod
    def active_weights(self, beta: float) -> int:
        """
        Method to get number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        pass