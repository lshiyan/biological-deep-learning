from abc import ABC
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces.layer import NetworkLayer


class HiddenLayer(NetworkLayer, ABC):
    #################################################################################################
    # Constructor Method
    #################################################################################################
    """
    INTERFACE
    Hidden layer in ANN -> All hidden layers should implement this class
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): the device that the module will be running on
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            id_tensor (torch.Tensor): id tensor of layer
    """
    def __init__(self, input_dimension: int, 
                 output_dimension: int, 
                 device: str,
                 learning_rate: float = 0.005,
                 lamb: float = 1,
                 gamma: float = 0.99, 
                 eps: float = 0.01) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            lamb: lambda hyperparameter for lateral inhibition
            learning_rate: how fast model learns at each iteration
            gamma: affects exponentialaverages updates
            eps: affects weight decay updates
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device, learning_rate)
        self.gamma: float = gamma
        self.lamb: float = lamb
        self.eps: float = eps
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension).to(self.device)
        self.id_tensor: torch.Tensor = self.create_id_tensors().to(self.device)



    #################################################################################################
    # Different Inhibition Methods
    #################################################################################################
    def _relu_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates ReLU lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        relu: nn.ReLU = nn.ReLU()
        input: torch.Tensor = relu(input)
        max_ele: int = torch.max(input).item()
        input = torch.pow(input, self.lamb)
        output: torch.Tensor =  input / abs(max_ele) ** self.lamb
        return output
    
    
    def _softmax_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates softmax lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        output: torch.Tensor = F.softmax(input, dim=-1)
        return output
    
    
    def _exp_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates exponential (Hopfield Networks) lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        max_ele: int = torch.max(input).item()
        output: torch.Tensor = F.softmax((input - max_ele)*self.lamb, dim=-1)
        return output
    
        
    def _wta_inhibition(self, input:torch.Tensor, top_k: int = 1) -> torch.Tensor:
        """
        METHOD
        Calculates winner-takes-all lateral inhibition
        @param
            input: input to layer
            top_k: 
        @return
            output: activation after lateral inhibition
        """
        # NOTE: this function does not work as of yet
        # Step 1: Flatten the tensor to apply top-k
        flattened_input = input.flatten()

        # Step 2: Get the top-k values and their indices
        topk_values, _ = torch.topk(flattened_input, top_k)
        threshold = topk_values[-1]

        # Step 3: Apply the threshold to keep only the top-k values
        output: torch.Tensor = torch.where(input >= threshold, input, torch.tensor(0.0, device=input.device))

        return output

    
    def _gaussian_inhibition(self, input: torch.Tensor, sigma=1.0) -> torch.Tensor:
        """
        METHOD
        Calculates gaussian lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        # NOTE: this does not work as of yet
        size: int = int(2 * sigma + 1)
        kernel: torch.Tensor = torch.tensor([torch.exp(-(i - size // 2) ** 2 / (2 * sigma ** 2)) for i in range(size)])
        kernel = kernel / torch.sum(kernel)

        output: torch.Tensor = F.conv1d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=size//2).squeeze(0).squeeze(0)
        return output

    
    
    #################################################################################################
    # Different Weight Updates Methods
    #################################################################################################
    def _linear_sanger_rule(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Sanger's Rules.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Retrieve initial weights (transposed) 
        initial_weight: torch.Tensor = torch.transpose(self.fc.weight.clone().detach().to(self.device), 0, 1)

        # Calculate Sanger's Rule
        A: torch.Tensor = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y).to(self.device)
        A = A * (y.unsqueeze(1))

        # Compute change in weights
        delta_weight: torch.Tensor = self.lr * (outer_prod - A)

        # Update the weights
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)


    def _orthogonal_rule(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Fully Orthogonal Rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Retrieve initial weights
        initial_weight: torch.Tensor = self.fc.weight.clone().detach().to(self.device)

        # Calculate Fully Orthogonal Rule
        ytw = torch.matmul(y.unsqueeze(0), initial_weight).to(self.device)
        norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0)).to(self.device)

        # Compute change in weights
        delta_weight: torch.Tensor = self.lr * (outer_prod - norm_term)

        # Update the weights
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)


    def _sigmoid_sanger_rule(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Sanger's Rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        sigmoid = nn.Sigmoid()
        
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)
        initial_weight: torch.Tensor = torch.transpose(self.fc.weight.clone().detach().to(self.device), 0, 1)
        self.id_tensor = self.id_tensor.to(self.device)
        self.exponential_average = self.exponential_average.to(self.device)
        A: torch.Tensor = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))
        delta_weight: torch.Tensor = self.lr * (outer_prod - A)
        self.fc.weight = nn.Parameter(self.fc.weight + delta_weight * sigmoid(self.fc.weight) * sigmoid(1 - self.fc.weight), requires_grad=False)
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)



    #################################################################################################
    # Activations and weight/bias updates that will be called for train/eval forward
    #################################################################################################
    def inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Choice of inhibition to be used.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            inputs after inhibition
        """
        raise NotImplementedError("This method has yet to be implemented.")


    def update_weights(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using a certain rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        raise NotImplementedError("This method has yet to be implemented.")
    

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Defines the way the biases will be updated at each iteration of the training
        @param
            output: The output tensor of the layer.
        @return
            None
        """
        raise NotImplementedError("This method has yet to be implemented.")
    
    
    
    #################################################################################################
    # Training/Evaluation during forward pass
    #################################################################################################
    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None) -> torch.Tensor:
        """
        METHOD
        Defines how an input data flows throw the network when training
        @param
            input: input data into the layer
            clamped_output: *NOT USED*
        @return
            output: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method has yet to be implemented.")
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Define how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            output: returns the data after passing it throw the layer
        """
        raise NotImplementedError("This method has yet to be implemented.")