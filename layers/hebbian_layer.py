import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from numpy import outer
from layers.layer import NetworkLayer
from layers.scheduler import Scheduler


"""
Class defining the functionality of the hebbian layer
"""
class HebbianLayer(NetworkLayer):
    """
    Constructor method NetworkLayer
    @param
        input_dimension (int) = number of inputs into the layer
        output_dimension (int) = number of outputs from layer
        lamb (float) = lambda hyperparameter for latteral inhibition
        heb_lr (float) = how fast model learns at each iteration
        gamma (float) = decay factor -> factor to decay learning rate
        eps (float) = to avoid division by 0
    @attr.
        PARENT ATTR.
            input_dimension (int) = number of inputs into the layer
            output_dimension (int) = number of outputs from layer
            device_id (int) = the device that the module will be running on
            lamb (float) = lambda hyperparameter for latteral inhibition
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            scheduler (layers.Scheduler) = scheduler for current layer
            eps (float) = to avoid division by 0
            exponential_average (torch.Tensor) = 0 tensor to keep track of exponential averages
            gamma (float) = decay factor -> factor to decay learning rate
            id_tensor (torch.Tensor) = id tensor of layer
        OWN ATTR.
    @return
        ___ (layers.Hebbianlayer) = returns instance of HebbianLayer
    """
    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, device_id, lamb, heb_lr, gamma, eps)


    """
    Calculates lateral inhibition
    @param
        x (torch.Tensor) = input to the ReLU function
    @return
        x (torch.Tensor) = activatin after lateral inhibition
    """
    def inhibition(self, x):
        relu = nn.ReLU()
        x = relu(x) # NOTE: Why are we using ReLU -> why not keep negative activations
        max_ele = torch.max(x).item()
        x = torch.pow(x, self.lamb)
        x /= abs(max_ele) ** self.lamb
        return x
        
    """
    Defines the way the weights will be updated at each iteration of the training.
    Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k).
    Calculates outer product of input and output and adds it to matrix.
    @param
        input (torch.Tensor) = the inputs into the layer
        output (torch.Tensor) = the output of the layer
        clamped_output (TODO: ???) = ???
    @return
        ___ (void) = no returns
    """
    # TODO: write out explicitly what each step of this method does
    def update_weights(self, input, output, clamped_output=None):
        x = input.clone().detach().float().squeeze().to(self.device_id)
        x.requires_grad_(False)
        y = output.clone().detach().float().squeeze().to(self.device_id)
        y.requires_grad_(False)
        
        # Move tensors to CPU before calling outer
        outer_prod = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy()))

        # Move back to GPU
        outer_prod = outer_prod.to(self.device_id)

        initial_weight = torch.transpose(self.fc.weight.clone().detach().to(self.device_id), 0, 1)

        # Ensure id_tensor and exponential_average are on the same device as the others
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)

        #print(f"UPDATE-WEIGHTS FUNCTION initial weight: {initial_weight.device}")
        #print(f"UPDATE-WEIGHTS FUNCTION id tensor: {self.id_tensor.device}")
        #print(f"UPDATE-WEIGHTS FUNCTION y: {y.device}")
        #Sprint(f"UPDATE-WEIGHTS FUNCTION exponential_average: {self.exponential_average.device}")

        A = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))
        delta_weight = self.alpha * (outer_prod - A)
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
    
    
    """
    Defines the way the weights will be updated at each iteration of the training
    @param
        output (torch.Tensor) = the output of the layer
    @return
        ___ (void) = no returns
    """
    # TODO: write out explicitly what each step of this method does     
    def update_bias(self, output):
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias)
        A = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)

    """
    Decays the overused weights and increases the underused weights using tanh functions.
    @param
    @return
        ___ (void) = no returns
    """
    # TODO: write out explicitly what each step of this method does
    def weight_decay(self):
        tanh = nn.Tanh()
        average = torch.mean(self.exponential_average).item()
        A = self.exponential_average / average
        growth_factor_positive = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative = torch.reciprocal(growth_factor_positive)
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")
    
    """
    Feed forward
    @param
        x (torch.Tensor) = input processed data
        clamped_output (???) = ???
    @retrun
        x (torch.Tensor) = data after going through hebbian layer
    """
    # NOTE: what does clamped_output mean?
    def forward(self, x, clamped_output=None):

        #print(f"STEP 1: INSIDE FORWARD OF HEBBIAN_LAYER (the device of x): {x.device}")


        input_copy = x.clone().to(self.device_id).float()

        #print(x.type())
        #print(self.fc.weight.type())
        #print(self.fc.bias.type())
        
        #print(f"STEP 2: INSIDE FORWARD OF HEBBIAN_LAYER -  (the device of x): {x.device}")
        x = x.to(self.device_id)
        #print(f"STEP 3.1: {self.device_id}")
        #print(f"STEP 3.2: {x.device}")

        x = self.fc(x)
        x = self.inhibition(x)

        #print(f"STEP 4: {x.device}")
        self.update_weights(input_copy, x, clamped_output)

        #self.update_bias(x)
        self.weight_decay() 
        return x

    
    """
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    @param
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self, result_path):
        row = 0
        col = 0

        root = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break

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
        
        file_path = result_path + '/hebbianlayerweights.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.show()
        


    """
    Counts the number of active feature selectors (above a certain cutoff beta).
    @param
        beta (float) = cutoff value determining which neuron is active and which is not
    @return
        ___ (void) = no returns
    """
    # TODO: define how active_weights should be counted in the hebbian layer
    def active_weights(self, beta):
        pass


    """
    Sets the scheduler for this layer
    @param
    @return
        ___ (void) = no returns
    """
    def set_scheduler(self, scheduler=None):
        self.scheduler = Scheduler(self.alpha, 1000, self.gamma) if scheduler == None else scheduler