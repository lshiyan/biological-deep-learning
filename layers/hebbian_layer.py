import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            device_id (int) = the device that the module will be running on
            lamb (float) = lambda hyperparameter for lateral inhibition (FOR relu_inhibition specifically)
            alpha (float) = how fast model learns at each iteration
            fc (fct) = function to apply linear transformation to incoming data
            eps (float) = to avoid division by 0
        OWN ATTR.
            exponential_average (torch.Tensor) = 0 tensor to keep track of exponential averages
            gamma (float) = decay factor -> factor to decay learning rate
            id_tensor (torch.Tensor) = id tensor of layer
    @return
        ___ (layers.Hebbianlayer) = returns instance of HebbianLayer
    """
    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):
        super ().__init__(input_dimension, output_dimension, device_id, lamb, heb_lr, eps)
        self.gamma = gamma
        self.exponential_average = torch.zeros(self.output_dimension)
        self.id_tensor = self.create_id_tensors()


    """
    This function mimicks the behaviour of classical hopfield net!

    This function implements a form of lateral inhibition with a ReLU activation function

    Lateral inhibition is a process where an activated neuron suppresses the activity of its neighbors, enhancing the contrast in the activity pattern.
    
    This function ensures that stronger activations are further emphasized
    This can lead to more distinct and sparse activations, which can be beneficial

    @param
        x (torch.Tensor) = input to the ReLU function
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def relu_inhibition(self, x):

        # Step 1: ReLU Activation
        relu = nn.ReLU()
        x = relu(x)

        # Step 2: Find max element -> this step finds the maximum value in the tensor x after ReLU activation
        max_ele = torch.max(x).item()
            # The .item() method converts the tensor containing the maximum value to a Python scalar.


        # Step 3: Apply power function
        x = torch.pow(x, self.lamb)
            # Raises each element of the tensor x to the power of self.lamb.
            # This operation enhances the differences between the activated values, with higher values becoming more pronounced.


        # Step 4: Normalization
        x /= abs(max_ele) ** self.lamb
            # Normalizes the tensor x by dividing it by the absolute value of the maximum element raised to the power of self.lamb.
            # This ensures that the scaled values are within a consistent range, preventing them from becoming excessively large or small.

        return x
    



    """
    This function implements the divisive normalization inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def divisive_normalization_inhibition(x, epsilon=1e-6):
        sum_activity = torch.sum(x)
        x = x / (sum_activity + epsilon)
        return x
    



    """
    This function implements the gaussian inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def gaussian_inhibition(x, sigma=1.0):
        size = int(2 * sigma + 1)
        kernel = torch.tensor([torch.exp(-(i - size // 2) ** 2 / (2 * sigma ** 2)) for i in range(size)])
        kernel = kernel / torch.sum(kernel)

        x = F.conv1d(x.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=size//2).squeeze(0).squeeze(0)
        return x
    


    """
    This function implements the winner takes all inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def winner_take_all_inhibition(x, top_k=1):
        topk_values, _ = torch.topk(x, top_k)
        threshold = topk_values[-1]
        x = torch.where(x >= threshold, x, torch.tensor(0.0, device=x.device))
        return x


    """
    This function implements the softmax inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def softmax_inhibition(x):
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)
        


    """
    Calculates lateral inhibition using an exponential function -> This function mimics MODERN HOPFIELD NET
    @param
        x (torch.Tensor)
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def modern_hopfield_inhibition(self, x):
        relu = nn.ReLU()
        x = relu(x)  # Apply ReLU activation to ensure non-negative values
        max_ele = torch.max(x).item()  # Find the maximum element in the tensor

        # Apply the exponential function to the activations with stability
        exp_x = torch.exp(self.lamb * (x - max_ele))

        # Normalize the activations by dividing by the sum of the exponentials
        x = exp_x / torch.sum(exp_x)
        
        return x




    """
    Defines the way the weights will be updated at each iteration of the training.
    Employs Sanger's Rule, deltaW_(ij)=alpha*x_j*y_i-alpha*y_i*sum(k=1 to i) (w_(kj)*y_k).
    Calculates outer product of input and output and adds it to matrix.
    @param
        input (torch.Tensor) = the inputs into the layer
        output (torch.Tensor) = the output of the layer
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        ___ (void) = no returns
    """
    def update_weights(self, input, output, clamped_output=None):
        
    # STEP 1: Extract and prepare input and output vectors so that I can update the weights accordingly
        x = input.clone().detach().float().squeeze().to(self.device_id)
        x.requires_grad_(False)
        y = output.clone().detach().float().squeeze().to(self.device_id)
        y.requires_grad_(False)

        # What is done in the above lines of code is the following:
            # The input (x) and output (y) tensors are cloned, detached from the computation graph, converted to float, squeezed to remove any singleton dimensions, and moved to the specified device (self.device_id)
            # requires_grad_(False) ensures these tensors will not track gradients.


    # STEP 2: Calculate the outer product
        outer_prod = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy()))  # Move tensors to CPU before calling outer
        outer_prod = outer_prod.to(self.device_id)
        # Here, the outer product of the output (y) and input (x) vectors is computed and then moved back to the GPU


    # STEP 3: Retrieve and Prepare initial weights
        initial_weight = torch.transpose(self.fc.weight.clone().detach().to(self.device_id), 0, 1)
        # Here, the current weights of the fully connected layer (self.fc.weight) are cloned, detached, and transposed. This prepares the weights for subsequent calculations.

        
    # STEP 4: Ensure Tensors are on the same device
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)
        # Here, I move the id_tensor and exponential_average tensors to the same device as the other tensors to ensure compatibility in calculations.


    # STEP 5: Compute lateral inhibition term
        A = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))  # 'A' represents the inhibitory effect based on current weights and activations.

    # TODO -> UNDERSTAND THIS EINSUM!


    # STEP 6: Compute weight update
        delta_weight = self.alpha * (outer_prod - A)
        # Here, the weight update (delta_weight) is computed by subtracting the lateral inhibition term (A) from the outer product and scaling by the learning rate self.alpha.

    # STEP 7: Update the weights
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        # The weights of the fully connected layer are updated by adding delta_weight. 
        # The updated weights are wrapped in nn.Parameter with requires_grad=False to ensure they do not track gradients.


    # STEP 8: Update the exponential average
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        # The exponential average of the output activations is updated using a moving average formula with decay factor self.gamma.
    
    
    """
    Defines the way the biases will be updated at each iteration of the training
    It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.
    The method applies an exponential decay to the biases, which is modulated by the output probabilities,
    and scales the update by the learning rate. 
    The biases are normalized after the update.
    @param
        output (torch.Tensor) = The output tensor of the layer.
    @return
        ___ (void) = no returns
    """
    def update_bias(self, output):
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)


    """
    Decays the overused weights and increases the underused weights using tanh functions.
    @param
    @return
        ___ (void) = no returns
    """
    def weight_decay(self):
        tanh = nn.Tanh()

        # Gets average of exponential averages
        average = torch.mean(self.exponential_average).item()

        # Gets ratio vs mean
        A = self.exponential_average / average

        # calculate the growth factors
        growth_factor_positive = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative = torch.reciprocal(growth_factor_positive)

        # Update the weights depending on growth factor
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")
    
    
    """
    Method that defines how an input data flows throw the network when training
    @param
        x (torch.Tensor) = input data into the layer
        clamped_output (torch.Tensor) = *NOT USED*
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the layer
    """
    def _train_forward(self, x, clamped_output=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        # Copy input -> calculate output -> update weights -> return output
        input_copy = x.clone().to(self.device_id).float()
        x = x.to(self.device_id)
        x = self.fc(x)
        x = self.modern_hopfield_inhibition(x)
        self.update_weights(input_copy, x)
        #self.update_bias(x)
        self.weight_decay()
        return x
    
    
    """
    Method that defines how an input data flows throw the network when testing
    @param
        x (torch.Tensor) = input data into the layer
    @return
        data_input (torch.Tensor) = returns the data after passing it throw the layer
    """
    def _eval_forward(self, x):
        # Copy input -> calculate output -> return output
        x = x.to(self.device_id)
        x = self.fc(x)
        x = self.modern_hopfield_inhibition(x)
        return x
    
    
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
    Visualizes the weight/features learnt by neurons in this layer using their heatmap
    @param
    @return
        ___ (void) = no returns
    """
    def visualize_weights(self, result_path, num, use):
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