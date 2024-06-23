import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from numpy import outer
from layers.layer import NetworkLayer


class HebbianLayer(NetworkLayer):
    """
    CLASS
    Defining the functionality of the hebbian layer
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device_id (str): the device that the module will be running on
            lamb (float): lambda hyperparameter for latteral inhibition
            alpha (float): how fast model learns at each iteration
            fc (nn.Linear): function to apply linear transformation to incoming data
            eps (float): to avoid division by 0
        OWN ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
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
        super().__init__(input_dimension, output_dimension, device_id, lamb, heb_lr, eps)
        self.gamma: float = gamma
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension)
        self.id_tensor: torch.Tensor = self.create_id_tensors()


    def inhibition(self, input: torch.Tensor):
        """
        METHOD
        Calculates lateral inhibition
        @param
            input: input to the ReLU function
        @return
            input: activation after lateral inhibition
        """
        relu: nn.ReLU = nn.ReLU()
        input: torch.Tensor = relu(input)
        max_ele: int = torch.max(input).item()
        input = torch.pow(input, self.lamb)
        input /= abs(max_ele) ** self.lamb
        return input
        

    def update_weights(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Update weights using Sanger's Rule.
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
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy()))  # Move tensors to CPU before calling outer

        # Move back to GPU
        outer_prod = outer_prod.to(self.device_id)

        # STEP 3: Retrieve and Prepare Initial Weights
            # Here, weight is of size -> output_dimensionXinput_dimension
        initial_weight: torch.Tensor = torch.transpose(self.fc.weight.clone().detach().to(self.device_id), 0, 1)

        # Ensure id_tensor and exponential_average are on the same device as the others
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)

        # STEP 4: COMPUTE LATERAL INHIBITION TERM
        A: torch.Tensor = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))

        # STEP 5: Compute weight update
        delta_weight: torch.Tensor = self.alpha * (outer_prod - A)

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
            input: returns the data after passing it throw the layer
        """
        # Copy input -> calculate output -> update weights -> return output
        input_copy = input.clone().to(self.device_id).float()
        input = input.to(self.device_id)
        input = self.fc(input)
        input = self.inhibition(input)
        self.update_weights(input_copy, input)
        #self.update_bias(input)
        self.weight_decay()
        return input
    

    def _eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Define how an input data flows throw the network when testing
        @param
            input: input data into the layer
        @return
            input: returns the data after passing it throw the layer
        """
        # Copy input -> calculate output -> return output
        input = input.to(self.device_id)
        input = self.fc(input)
        input = self.inhibition(input)
        return input
    

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
        row = 0
        col = 0

        root = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Get the weights and create heatmap
        weight = self.fc.weight
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
            
            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device_id)
        
        file_path = result_path + f'/hebbian/hebbianlayerweights-{num}-{use}.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()