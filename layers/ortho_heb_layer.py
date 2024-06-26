import math
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.figure
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import outer
import numpy as np
from layers.layer import NetworkLayer


class OrthoHebLayer(NetworkLayer):
    """
    CLASS
    Defining the functionality of the orthogonal hebbian layer
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device_id (str): the device that the module will be running on
            alpha (float): how fast model learns at each iteration
            fc (nn.Linear): function to apply linear transformation to incoming data
        OWN ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for latteral inhibition
            eps (float): to avoid division by 0
            id_tensor (torch.Tensor): id tensor of layer
    """
    def __init__(self, input_dimension: int, 
                 output_dimension: int, 
                 device_id: str, 
                 lamb: float = 1, 
                 heb_lr: float = 0.005, 
                 gamma: float = 0.99, 
                 eps: float = 0.01) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            lamb: lambda hyperparameter for latteral inhibition
            heb_lr: how fast model learns at each iteration
            gamma: decay factor -> factor to decay learning rate
            eps: to avoid division by 0
        @return
            None
        """
        super().__init__(input_dimension, output_dimension, device_id, heb_lr)
        self.gamma: float = gamma
        self.lamb = lamb
        self.eps = eps
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension)
        self.id_tensor: torch.Tensor = self.create_id_tensors()


    def relu_inhibition(self, input: torch.Tensor) -> torch.Tensor:
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
    
    
    def softmax_inhibition(self, input: torch.Tensor) -> torch.Tensor:
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
    
    
    def exp_inhibition(self, input: torch.Tensor) -> torch.Tensor:
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
    
        
    def wta_inhibition(self, input:torch.Tensor, top_k: int = 1) -> torch.Tensor:
        """
        METHOD
        Calculates winner-takes-all lateral inhibition
        @param
            input: input to layer
            top_k: 
        @return
            output: activation after lateral inhibition
        """
        # Step 1: Flatten the tensor to apply top-k
        flattened_input = input.flatten()

        # Step 2: Get the top-k values and their indices
        topk_values, _ = torch.topk(flattened_input, top_k)
        threshold = topk_values[-1]

        # Step 3: Apply the threshold to keep only the top-k values
        output: torch.Tensor = torch.where(input >= threshold, input, torch.tensor(0.0, device=input.device))

        return output


    def gaussian_inhibition(input: torch.Tensor, sigma=1.0) -> torch.Tensor:
        """
        METHOD
        Calculates gaussian lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """
        size: int = int(2 * sigma + 1)
        kernel: torch.Tensor = torch.tensor([torch.exp(-(i - size // 2) ** 2 / (2 * sigma ** 2)) for i in range(size)])
        kernel = kernel / torch.sum(kernel)

        output: torch.Tensor = F.conv1d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=size//2).squeeze(0).squeeze(0)
        return output


    def update_weights(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Fully Orthogonal Rule.
        @param
            input: the inputs into the layer
            output: the output of the layer
        @return
            None
        """
        # STEP 1: Extract and prepare input and output vectors to be 'x' and 'y' respectively
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device_id)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device_id)
        y.requires_grad_(False)
        
        # STEP 2: Calculate the Outer product of 'x' and 'y'
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device_id)

        # STEP 3: Retrieve and Prepare Initial Weights
        initial_weight: torch.Tensor = self.fc.weight.clone().detach().to(self.device_id)

        # Ensure id_tensor and exponential_average are on the same device as the others
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)

        # STEP 4: Fully orthogonal rule
        ytw = torch.matmul(y.unsqueeze(0), initial_weight).to(self.device_id)
        norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0)).to(self.device_id)

        # STEP 5: Compute weight update
        delta_weight: torch.Tensor = self.alpha * (outer_prod - norm_term)

        # STEP 6: Update the weights
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)

        # STEP 7: Update the exponential average
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
    

    def update_bias(self, output: torch.Tensor) -> None:
        """
        METHOD
        Defines the way the biases will be updated at each iteration of the training
        It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.
        The method applies an exponential decay to the biases, which is modulated by the output probabilities,
        and scales the update by the learning rate. 
        The biases are normalized after the update.
        @param
            output: The output tensor of the layer.
        @return
            None
        """
        y: torch.Tensor = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A: torch.Tensor = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes: float = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
        

    def weight_decay(self) -> None:
        """
        METHOD
        Decays the overused weights and increases the underused weights using tanh functions.
        @param
            None
        @return
            None
        """
        tanh: nn.Tanh = nn.Tanh()

        # Gets average of exponential averages
        average: float = torch.mean(self.exponential_average).item()

        # Gets ratio vs mean
        A: torch.Tensor = self.exponential_average / average

        # calculate the growth factors
        growth_factor_positive: torch.Tensor = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative: torch.Tensor = torch.reciprocal(growth_factor_positive)

        # Update the weights depending on growth factor
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")
    

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
        # Copy input -> calculate output -> update weights -> return output
        input_copy = input.clone().to(self.device_id).float()
        input = input.to(self.device_id)
        input = self.fc(input)
        output = self.relu_inhibition(input)
        # output = self.softmax_inhibition(input)
        # output = self.exp_inhibition(input)
        # output = self.wta_inhibition(input)
        # output = self.norm_inhibition(input)
        # output = self.gaussian_inhibition(input)
        self.update_weights(input_copy, output)
        #self.update_bias(input)
        self.weight_decay()
        return output
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Define how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            output: returns the data after passing it throw the layer
        """
        # Copy input -> calculate output -> return output
        input = input.to(self.device_id)
        input = self.fc(input)
        output = self.relu_inhibition(input)
        # output = self.softmax_inhibition(input)
        # output = self.exp_inhibition(input)
        # output = self.wta_inhibition(input)
        # output = self.norm_inhibition(input)
        # output = self.gaussian_inhibition(input)
        return output
    

    def visualize_weights(self, result_path: str, num: int, use: str) -> None:
        """
        METHOD
        Vizualize the weight/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
        @return
            None
        """
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
            random_feature_selector = random_feature_selector.to(self.device_id)
        
        file_path: str = result_path + f'/hebbian/hebbianlayerweights-{num}-{use}.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()