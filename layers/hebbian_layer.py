import torch
import torch.nn as nn
from numpy import outer
from layers.layer import NetworkLayer


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
            lamb (float) = lambda hyperparameter for latteral inhibition
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            scheduler (layers.Scheduler) = scheduler for current layer
            eps (float) = to avoid division by 0
            exponential_average (torch.Tensor) = 0 tensor to keep track of exponential averages
            gamma (float) = decay factor -> factor to decay learning rate
            id_tensor (torch.Tensor) = id tensor of layer
        OWN ATTR.
    """
    def __init__(self, input_dimension, output_dimension, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, lamb, heb_lr, gamma, eps)

    """
    Calculates latheral inhibition
    @param
        x (torch.Tensor) = input to the ReLU function
    """
    def inhibition(self, x):
        x = self.relu(x) 
        max_ele = torch.max(x).item()
        x = torch.pow(x, self.lamb)
        x /= abs(max_ele) ** self.lamb
        return x
        
    """
    Defines the way the weights will be updated at each iteration of the training.
    Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k).
    Calculates outer product of input and output and adds it to matrix.
    @param
        input (???) = ???
        output (???) = ???
    """
    # TODO: write out explicitly what each step of this method does
    # TODO: finish documentation when understand
    def update_weights(self, input, output, clamped_output=None):
        x = torch.tensor(input.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
        y = torch.tensor(output.clone().detach(), requires_grad=False, dtype=torch.float).squeeze()
        outer_prod = torch.tensor(outer(y, x))
        initial_weight = torch.transpose(self.fc.weight.clone().detach(), 0,1)
        A = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))
        delta_weight = self.alpha * (outer_prod - A)
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        self.exponential_average = torch.add(self.gamma * self.exponential_average,(1 - self.gamma) * y)
    
    """
    Defines the way the weights will be updated at each iteration of the training
    @param
        input (???) = ???
        output (???) = ???
    """
    # TODO: write out explicitly what each step of this method does
    # TODO: finish documentation when understand        
    def update_bias(self, output):
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias)
        A = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)

    """
    Decays the overused weights and increases the underused weights using tanh functions.
    """
    # TODO: write out explicitly what each step of this method does
    def weight_decay(self):
        average = torch.mean(self.exponential_average).item()
        A = self.exponential_average / average
        growth_factor_positive = self.eps * self.tanh(-self.eps * (A - 1)) + 1
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
    """
    def forward(self, x, clamped_output=None):
        input_copy = x.clone()
        x = self.fc(x)
        x = self.inhibition(x)
        self.update_weights(input_copy, x, clamped_output)
        #self.update_bias(x)
        self.weight_decay() 
        return x

    
    """
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    @param
        row (int) = number of rows in display
        col (int) = number of columns in display
    """
    def visualize_weights(self, row, col):
        weight = self.fc.weight
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector = weight[ele]
            heatmap = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))),
                                                    int(math.sqrt(self.fc.weight.size(1))))
            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
        plt.tight_layout()
        plt.show()